evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],  # valid, test 공용으로 사용
    save_best='mean_class_accuracy')

# optimizer
lr=1e-3
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
work_dir = './work_dirs/train_output/hello/cdar/tsm'
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_100e_kinetics400_rgb/tsm_r50_1x1x8_100e_kinetics400_rgb_20210701-7ff22268.pth'
ckpt_revise_keys = []#[('cls_head', 'unusedhead')]
