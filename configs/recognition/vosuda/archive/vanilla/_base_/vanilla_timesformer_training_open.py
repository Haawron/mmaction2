evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'H_mean_class_accuracy', 'mean_class_accuracy', 'confusion_matrix'],  # valid, test 공용으로 사용
    save_best='H_mean_class_accuracy')

# optimizer
lr=.1
optimizer = dict(
    type='SGD',
    lr=.1,
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
total_epochs = 300
work_dir = './work_dirs/hello/timesformer/vanilla/'
load_from = None#'data/weights/timesformer/timesformer_8x32_224_howto100M_mmaction.pth'
