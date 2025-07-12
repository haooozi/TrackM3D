import torch
import torch.nn as nn

class KfMask(nn.Module):
    def __init__(self, height, width, stride):
        super(KfMask, self).__init__()
        self.H = height
        self.W = width
        self.S = stride
        self.mask = torch.zeros(self.batch, self.H, self.W, 1) # [B, H, W, 1]

    @torch.no_grad()
    def forward(self, motion): # motion [B, 2]
        B, _ = motion.size()
        h_start = torch.ceil(self.H // 4 + motion[:, 0:1]).long().unsqueeze(-1) # [B, 1]
        h_end = torch.ceil(3 * self.H // 4 + motion[:, 0:1]).long().unsqueeze(-1)
        w_start = torch.ceil(self.W // 4 + motion[:, 1:]).long().unsqueeze(-1)
        w_end = torch.ceil(3 * self.W // 4 + motion[:, 1:]).long().unsqueeze(-1)

        x_range = torch.arange(self.W).view(1, 1, self.W).expand(B, self.H, self.W)
        y_range = torch.arange(self.H).view(1, self.H, 1).expand(B, self.H, self.W)

        mask_x = (x_range >= h_start) & (x_range <= h_end)

        mask_y = (y_range >= w_start) & (y_range <= w_end)

        mask = (mask_x & mask_y).unsqueeze(-1)

        self.mask[mask] = 1

        return self.mask

if __name__ == '__main__':
    kfm = KfMask(64, 16, 16, 128 // 16)
    motion = torch.rand([64, 2])
    output = kfm(motion)
    print(output.shape)