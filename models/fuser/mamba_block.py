import torch
import torch.nn as nn
import math
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from timm.models.layers import DropPath
from torch import Tensor
from typing import Optional


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **{}, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

# if __name__ == "__main__":
#     Mamba_fuser = MixerModel(d_model=128,
#                              n_layer=6,
#                              rms_norm=False,
#                              drop_out_in_block=0.,
#                              drop_path=0.1).cuda()
#     input = torch.randn([4, 256, 128]).cuda() # [B, Tokens, Channels]
#     # pos = torch.randn([4, 256, 128]).cuda() # [B, Tokens, Channels]
#     output = Mamba_fuser(input)
#     print(output.shape)

def counterclockwise_order():
    data = torch.tensor(
           [[225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212,
             211, 210],
            [226, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157,
             156, 209],
            [227, 170, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110,
             155, 208],
            [228, 171, 122,  81,  80,  79,  78,  77,  76,  75,  74,  73,  72, 109,
             154, 207],
            [229, 172, 123,  82,  49,  48,  47,  46,  45,  44,  43,  42,  71, 108,
             153, 206],
            [230, 173, 124,  83,  50,  25,  24,  23,  22,  21,  20,  41,  70, 107,
             152, 205],
            [231, 174, 125,  84,  51,  26,   9,   8,   7,   6,  19,  40,  69, 106,
             151, 204],
            [232, 175, 126,  85,  52,  27,  10,   1,   0,   5,  18,  39,  68, 105,
             150, 203],
            [233, 176, 127,  86,  53,  28,  11,   2,   3,   4,  17,  38,  67, 104,
             149, 202],
            [234, 177, 128,  87,  54,  29,  12,  13,  14,  15,  16,  37,  66, 103,
             148, 201],
            [235, 178, 129,  88,  55,  30,  31,  32,  33,  34,  35,  36,  65, 102,
             147, 200],
            [236, 179, 130,  89,  56,  57,  58,  59,  60,  61,  62,  63,  64, 101,
             146, 199],
            [237, 180, 131,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,
             145, 198],
            [238, 181, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
             144, 197],
            [239, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
             195, 196],
            [240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
             254, 255]])
    sorted_values, indices = torch.sort(data.view(-1))
    re_sorted_values, re_indices = torch.sort(indices)
    return indices[None,:].cuda(), re_indices[None,:].cuda()

def gaussion_order():
    data = torch.tensor(
        [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
         1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667,
         0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.7333, 0.7333, 0.7333, 0.7333, 0.7333, 0.7333,
         0.7333, 0.7333, 0.7333, 0.7333, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000,
         0.6000, 0.6000, 0.6000, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.4667, 0.4667, 0.4667, 0.4667, 0.4667,
         0.4667, 0.4667, 0.4667, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.4667, 0.3333, 0.3333, 0.3333, 0.3333,
         0.3333, 0.3333, 0.4667, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.4667, 0.3333, 0.2000, 0.2000, 0.2000,
         0.2000, 0.3333, 0.4667, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.4667, 0.3333, 0.2000, 0.0667, 0.0667,
         0.2000, 0.3333, 0.4667, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.4667, 0.3333, 0.2000, 0.0667, 0.0667,
         0.2000, 0.3333, 0.4667, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.4667, 0.3333, 0.2000, 0.2000, 0.2000,
         0.2000, 0.3333, 0.4667, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.4667, 0.3333, 0.3333, 0.3333, 0.3333,
         0.3333, 0.3333, 0.4667, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.4667, 0.4667, 0.4667, 0.4667, 0.4667,
         0.4667, 0.4667, 0.4667, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000,
         0.6000, 0.6000, 0.6000, 0.6000, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.7333, 0.7333, 0.7333, 0.7333, 0.7333, 0.7333, 0.7333,
         0.7333, 0.7333, 0.7333, 0.7333, 0.7333, 0.8667, 1.0000],
        [1.0000, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667,
         0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
         1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
    tensor = data.view(-1)
    sorted_values, sorted_indices = torch.sort(tensor)
    unique_values, unique_indices = torch.unique(sorted_values, return_inverse=True)
    for idx in range(len(unique_values)):
        indices_same_value = (unique_indices == idx).nonzero().squeeze(1)
        random_indices = torch.randperm(len(indices_same_value))
        sorted_indices[indices_same_value] = sorted_indices[indices_same_value][random_indices]
    indices = sorted_indices
    re_sorted_values, re_indices = torch.sort(sorted_indices)
    return indices[None,:].cuda(), re_indices[None,:].cuda()

def gaussian(W, L, sigma):
    x = torch.linspace(-W//2, L//2, W)
    y = torch.linspace(-W//2, L//2, L)
    X, Y = torch.meshgrid(x, y)
    return torch.exp(-(X**2 + Y**2) / (2 * sigma**2)).cuda()

