_base_ = '../../default_runtime.py'
data_dir = '/data/waymo'
category_name = 'Vehicle'
batch_size = 128
point_cloud_range = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5] # not use
box_aware = True
use_rot = True

model = dict(
    type='TrackM3D',
    backbone=dict(type='VoxelNet',
                  points_features=3,
                  point_cloud_range=point_cloud_range,
                  voxel_size=[0.075, 0.075, 0.15],
                  grid_size=[21, 128, 128],
                  output_channels=128
                  ),
    fuser=dict(
        type='MambaFuser',
        d_model=256,
        n_layer=6,
        rms_norm=False,
        drop_out_in_block=0.,
        drop_path=0.1
    ),
    head=dict(
        type='MotionHead',
        q_distribution='gaussian',  # ['laplace', 'gaussian']
        use_rot=use_rot,
        box_aware=box_aware
    ),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
        post_processing=False,
        use_rot=use_rot
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='WaymoDataset',
        path=data_dir,
        category_name=category_name,
        mode='all'
    ),
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)
