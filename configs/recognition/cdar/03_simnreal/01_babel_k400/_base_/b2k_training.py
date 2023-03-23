evaluation = dict(
    interval=10,
    metrics=['kmeans', 'gcd_v2'],
    metric_options={'num_old_classes': 12, 'num_all_classes': 27},
    rule='greater',
    save_best='kmeans_balanced')

# optimizer
lr=1e-2
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
total_epochs = 50
work_dir = './work_dirs/train_output/hello/cdar'
load_from = 'data/weights/timesformer/timesformer_8x32_224_howto100m_mmaction.pyth'
