import torch
import torch.nn as nn
import torch.nn.functional as F
from .rle_loss import RLELoss
from mmengine.registry import MODELS
from mmcv.cnn import ConvModule

@MODELS.register_module()
class MotionHead(nn.Module):

    def __init__(self, q_distribution, use_rot=False, box_aware=False):
        super().__init__()
        self.use_rot = use_rot
        self.box_aware = box_aware
        norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
        self.ds_conv = nn.Sequential(
            # down-sample
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 512, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )
        self.regression_head = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.SyncBatchNorm(512, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.SyncBatchNorm(256, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(256, 128, bias=False),
            nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        self.criterion = RLELoss(q_distribution=q_distribution)

        if use_rot:
            self.rotation_head = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.SyncBatchNorm(512, eps=1e-3, momentum=0.01),
                nn.ReLU(True),
                nn.Linear(512, 256, bias=False),
                nn.SyncBatchNorm(256, eps=1e-3, momentum=0.01),
                nn.ReLU(True),
                nn.Linear(256, 128, bias=False),
                nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
                nn.ReLU(True),
                nn.Linear(128, 1)
            )

        if box_aware:
            self.wlh_mlp = nn.Sequential(
                nn.Linear(3, 128),
                nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
                nn.ReLU(True),
                nn.Linear(128, 512)
            )

    def forward(self, feats, wlh=None):
        feats = self.ds_conv(feats)
        if self.box_aware:
            wlh = self.wlh_mlp(wlh)
            feats = feats + wlh
        res = self.regression_head(feats)
        results = {
            'coors': res[:, :3],
            'sigma': res[:, 3:],
        }
        if self.use_rot:
            rot = self.rotation_head(feats.detach())
            results.update({'rotation': rot})
        return results

    def loss(self, results, data_samples):
        losses = dict()
        pred_coors = results['coors']
        sigma = results['sigma']
        gt_coors = torch.stack(data_samples['box_label'])
        losses['regression_loss'] = self.criterion(pred_coors, sigma, gt_coors)

        if self.use_rot:
            pred_rot = results['rotation']
            gt_rot = torch.stack(data_samples['theta'])
            losses['rotation_loss'] = F.smooth_l1_loss(pred_rot, gt_rot)

        return losses
