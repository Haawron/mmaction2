evaluation = dict(
    interval=5,
    metrics=['sskmeans'],  # valid, test 공용으로 사용
    metric_options=dict(sskmeans=dict(fixed_k=22, n_tries=100)),
    # ckpt-saving options
    save_best='sskmeans', rule='greater')

# optimizer
lr=.1
optimizer = dict(
    type='SGD',
    lr=lr,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=5e-5,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=lr*1e-3)
total_epochs = 200
work_dir = './work_dirs/hello/ucf2hmdb/svt/gcd4da/'
load_from = 'data/weights/svt/releases/download/v1.0/SVT_mmaction.pth'
