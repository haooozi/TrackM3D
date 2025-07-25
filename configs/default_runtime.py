default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=20,
                    save_best='precision', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = dict(type='Visualizer',
                  vis_backends=[dict(type='LocalVisBackend'),
                                dict(type='TensorboardVisBackend')])

epoch_num = 400
lr = 0.0001

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.00001),
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='DefaultOptimWrapperConstructor',
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.)
)

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()

custom_imports = dict(
    imports=['models', 'datasets'],
    allow_failed_imports=False
)


