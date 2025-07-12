import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, qk_feature_dim):
        super(CrossAttention, self).__init__()

        self.temp = qk_feature_dim ** -0.5
        self.WQ = nn.Linear(feature_dim, qk_feature_dim, bias=False)
        self.WK = nn.Linear(feature_dim // 2, qk_feature_dim, bias=False)
        self.WV = nn.Linear(feature_dim // 2, feature_dim, bias=False)

        # Init weights
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None):
        # [B, H*W, C] for q
        # [B, 5*5, C] for k, v

        w_q = self.WQ(query) # [B, H*W, D]
        w_q = F.normalize(w_q, p=2, dim=-1) # [B, H*W, D]

        w_k = self.WK(key) # [B, 5*5, D]
        w_k = F.normalize(w_k, p=2, dim=-1) # [B, 5*5, D]
        w_k = w_k.permute(0, 2, 1) # [B, D, 5*5]

        dot_prod = torch.bmm(w_q, w_k)  # [B, H*W, 5*5]

        affinity = F.softmax(dot_prod * self.temp, dim=-1) # [B, H*W, 5*5]

        w_v = self.WV(value) # [B, 5*5, D]
        w_v = F.normalize(w_v, p=2, dim=-1)  # [B, 5*5, D]

        output = torch.bmm(affinity, w_v) # [B, H*W, D]

        return output + query


class TopKCrossAttention(nn.Module):
    def __init__(self, feature_dim, qk_feature_dim, gate_value, topk):
        super(TopKCrossAttention, self).__init__()

        self.temp = qk_feature_dim ** -0.5
        self.WQ = nn.Linear(feature_dim, qk_feature_dim, bias=False)
        self.WK = nn.Linear(feature_dim // 2, qk_feature_dim, bias=False)
        self.WV = nn.Linear(feature_dim // 2, feature_dim, bias=False)

        self.gate_value = gate_value
        self.topk = topk

        # Init weights
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None):
        # [B, H*W, C] for q
        # [B, 5*5, C] for k, v
        w_q = self.WQ(query)  # [B, H*W, D]
        w_q = F.normalize(w_q, p=2, dim=-1)  # [B, H*W, D]

        w_k = self.WK(key)  # [B, 5*5, D]
        w_k = F.normalize(w_k, p=2, dim=-1)  # [B, 5*5, D]
        w_k = w_k.permute(0, 2, 1)  # [B, D, 5*5]

        dot_prod = torch.bmm(w_q, w_k)  # [B, H*W, 5*5]

        max_values, _ = torch.max(dot_prod, dim=2)

        mask = max_values > self.gate_value # [B, H*W, 1]

        affinity = F.softmax(dot_prod * self.temp, dim=-1)  # [B, H*W, 5*5]
        topk_values, topk_indices = torch.topk(affinity, k=self.topk, dim=2)

        w_v = self.WV(value)  # [B, 5*5, D]
        w_v = F.normalize(w_v, p=2, dim=-1)  # [B, 5*5, D]

        expanded_indices = topk_indices.permute(0, 2, 1)
        selected_wv = w_v.gather(dim=1, index=expanded_indices)
        gather = torch.bmm(topk_values, selected_wv)
        output = gather * mask[:, :, None]

        return output + query


if __name__ == '__main__':
    ca = CrossAttention(256, 64)
    query = torch.randn(1, 256, 256)
    key = torch.randn(1, 25, 128)
    value = key
    output = ca(query, key, value)
    print(output.shape)

    tca = TopKCrossAttention(256, 64, 0.2, 5)
    query = torch.randn(1, 256, 256)
    key = torch.randn(1, 25, 128)
    value = key
    output = tca(query, key, value)
    print(output.shape)

    flop = FlopCountAnalysis(model=tca, inputs=(query, key, value))
    print(flop.total())
    print(parameter_count_table(tca))

    flop = FlopCountAnalysis(model=ca, inputs=(query, key, value))
    print(flop.total())
    print(parameter_count_table(ca))