import torch
from mmengine.model import BaseModel
from datasets.metrics import estimateOverlap, estimateAccuracy
import numpy as np
from datasets import points_utils
from nuscenes.utils import geometry_utils
from mmengine.registry import MODELS

import torch


class KalmanFilter:
    def __init__(self, initial_position):
        # state vector [x, y, vx, vy]
        self.state = torch.tensor([[initial_position[0]], [initial_position[1]], [0.], [0.]])

        # F (4x4)
        self.F = torch.eye(4)
        self.F[0, 2] = 1  # x speed
        self.F[1, 3] = 1  # y speed

        #  H (2x4)
        self.H = torch.zeros(2, 4)
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y

        # Q (4x4)
        self.Q = torch.eye(4) * 0.01

        # R (2x2)
        self.R = torch.eye(2) * 0.1

        # P (4x4)
        self.P = torch.eye(4)

    def predict(self):
        self.state = torch.mm(self.F, self.state)
        self.P = torch.mm(self.F, torch.mm(self.P, self.F.T)) + self.Q

    def update(self, measurement):
        S = torch.mm(self.H, torch.mm(self.P, self.H.T)) + self.R
        K = torch.mm(self.P, torch.mm(self.H.T, torch.linalg.inv(S)))

        y = measurement - torch.mm(self.H, self.state)
        self.state = self.state + torch.mm(K, y)

        I = torch.eye(4)
        self.P = torch.mm((I - torch.mm(K, self.H)), self.P)

    def get_position(self):
        return self.state[0, 0].item(), self.state[1, 0].item()


@MODELS.register_module()
class TrackM3D(BaseModel):

    def __init__(self,
                 backbone=None,
                 fuser=None,
                 head=None,
                 cfg=None):
        super().__init__()
        self.config = cfg
        self.backbone = MODELS.build(backbone)
        self.fuse = MODELS.build(fuser)
        self.head = MODELS.build(head)
        self.kf = KalmanFilter(initial_position=[0,0])

    def forward(self,
                inputs,
                data_samples=None,
                motions=None,
                mode: str = 'predict',
                **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples, motions)
        elif mode == 'predict':
            return self.predict(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def get_feats(self, inputs, motion):
        prev_points = inputs['prev_points']
        this_points = inputs['this_points']

        prev_range = inputs['prev_range']
        this_range = inputs['this_range']

        stack_ranges = prev_range + this_range
        stack_points = prev_points + this_points

        stack_feats = self.backbone(stack_points, stack_ranges)
        cat_feats = self.fuse(stack_feats, motion)
        if self.config.box_aware:
            wlh = torch.stack(inputs['wlh']) if isinstance(inputs['wlh'], list) \
                else inputs['wlh'].unsqueeze(0)
            results = self.head(cat_feats, wlh)
        else:
            results = self.head(cat_feats)

        return results

    def inference(self, inputs, motion):
        results = self.get_feats(inputs, motion)
        coors = results['coors'][0]
        if self.config.use_rot:
            rot = results['rotation'][0]
            return coors, rot
        return coors

    def loss(self, inputs, data_samples, motions):
        results = self.get_feats(inputs, motions)
        losses = dict()
        losses.update(self.head.loss(results, data_samples))

        return losses

    def predict(self, inputs):
        ious = []
        distances = []
        results_bbs = []
        motions = []
        measurements = []
        for frame_id in range(len(inputs)):  # tracklet
            this_bb = inputs[frame_id]["3d_bbox"]

            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                last_coors = np.array([0., 0.])
                measurements.append(torch.tensor([0., 0.], dtype=torch.float32).cuda())
                motions.append(torch.tensor([0., 0.], dtype=torch.float32).cuda())
            else:
                data_dict, ref_bb, flag = self.build_input_dict(inputs, frame_id, results_bbs)
                if flag:
                    if self.config.use_rot:
                        coors, rot = self.inference(data_dict, motions[-1])
                        rot = float(rot)
                    else:
                        coors = self.inference(data_dict, motions[-1])
                        rot = 0.
                    coors_x = float(coors[0])
                    coors_y = float(coors[1])
                    coors_z = float(coors[2])
                    last_coors = np.array([coors_x, coors_y])
                    candidate_box = points_utils.getOffsetBB(
                        ref_bb, [coors_x, coors_y, coors_z, rot],
                        degrees=True, use_z=True, limit_box=False)

                    self.kf.predict()
                    measurement = measurements[-1]
                    self.kf.update(measurement)
                    x, y = self.kf.get_position()
                    motions.append(torch.tensor([x,y]))
                    measurements.append(torch.tensor([coors_x, coors_y]))

                else:
                    candidate_box = points_utils.getOffsetBB(
                        ref_bb, [last_coors[0], last_coors[1], 0, 0],
                        degrees=True, use_z=True, limit_box=False)

                    self.kf.predict()
                    measurement = measurements[-1]
                    self.kf.update(measurement)
                    x, y = self.kf.get_position()
                    motions.append(torch.tensor([x, y]))
                    measurements.append(torch.tensor([last_coors[0], last_coors[1]]))

                results_bbs.append(candidate_box)
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            ious.append(this_overlap)
            distances.append(this_accuracy)

        return ious, distances

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]

        prev_pc = prev_frame['pc']
        this_pc = this_frame['pc']
        ref_box = results_bbs[-1]

        range_prev = [-ref_box.wlh[1], -ref_box.wlh[0], -ref_box.wlh[2] * 0.75,
                      ref_box.wlh[1], ref_box.wlh[0], ref_box.wlh[2] * 0.75]
        prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, ref_box, range_prev)
        range_this = [-ref_box.wlh[1], -ref_box.wlh[0], -ref_box.wlh[2] * 0.75,
                      ref_box.wlh[1], ref_box.wlh[0], ref_box.wlh[2] * 0.75]
        this_frame_pc = points_utils.crop_pc_in_range(this_pc, ref_box, range_this)

        prev_points = prev_frame_pc.points.T
        this_points = this_frame_pc.points.T

        if self.config.post_processing is True:
            ref_bb = points_utils.transform_box(ref_box, ref_box)
            prev_idx = geometry_utils.points_in_box(ref_bb, prev_points.T, 1.25)
            if sum(prev_idx) < 3 and this_points.shape[0] < 25 and frame_id < 15:
                # not enough points for tracking
                flag = False
            else:
                flag = True
        else:
            flag = True

        if prev_points.shape[0] < 1:
            prev_points = np.zeros((1, 3), dtype='float32')
        if this_points.shape[0] < 1:
            this_points = np.zeros((1, 3), dtype='float32')


        data_dict = {'prev_points': [torch.as_tensor(prev_points, dtype=torch.float32).cuda()],
                     'this_points': [torch.as_tensor(this_points, dtype=torch.float32).cuda()],
                     'wlh': torch.as_tensor(ref_box.wlh, dtype=torch.float32).cuda(),
                     'prev_range': [torch.as_tensor(range_prev, dtype=torch.float32).cuda()],
                     'this_range': [torch.as_tensor(range_this, dtype=torch.float32).cuda()]
                     }

        return data_dict, results_bbs[-1], flag
