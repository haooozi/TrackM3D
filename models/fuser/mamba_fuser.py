import torch
import torch.nn as nn
from mmengine.registry import MODELS
from .mamba_block import MixerModel, counterclockwise_order

@MODELS.register_module()
class MambaFuser(nn.Module):
    def __init__(self, d_model, n_layer, drop_out_in_block, drop_path, rms_norm=False):
        super().__init__()
        self.mamba_co = MixerModel(d_model, n_layer, drop_out_in_block, drop_path, rms_norm=False)
        self.co, self.re_co = counterclockwise_order()

    def forward(self, stack_feats):
        B, C, H, W = stack_feats.size() # B=2*batchsize
        prev_feats, this_feats = torch.split(stack_feats, B // 2, 0)
        feats = torch.cat((prev_feats, this_feats), 1) #[batchsize, 2C, H, W]
        feats = feats.view(B // 2, 2 * C, -1) # [B, Channels, Tokens] Tokens=H*W

        index_co = torch.tile(self.co, (B // 2, 1)).unsqueeze(1) # [B, 1, H*W]
        feats_co = torch.gather(feats, dim=2, index=index_co.long().repeat(1, 2 * C, 1)) # [B, 2C, H*W]
        feats_co = feats_co.permute(0, 2, 1) # [B, H*W, 2C]
        feats_mco = self.mamba_co(feats_co) # [B, H*W, 2C]
        feats_mco = feats_mco.permute(0, 2, 1) # [B, 2C, H*W]
        index_re_co = torch.tile(self.re_co, (B // 2, 1)).unsqueeze(1)  # [B, 1, H*W]
        feats_re_co = torch.gather(feats_mco, dim=2, index=index_re_co.long().repeat(1, 2 * C, 1))  # [B, 2C, H*W]
        feats_re_co = feats_re_co.view(B // 2, 2 * C, H, W)  # [B, Channels, H, W]

        return feats_re_co


