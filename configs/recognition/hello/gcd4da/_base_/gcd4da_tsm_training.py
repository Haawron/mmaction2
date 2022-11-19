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
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    lr=lr,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=lr*1e-3)
total_epochs = 200
work_dir = './work_dirs/hello/ucfhmdb/tsm/gcd4da/'
load_from = 'work_dirs/train_output/hmdb2ucf/tsm/vanilla/source-only/4382__vanilla-tsm-hmdb2ucf-source-only/2/20220728-205928/best_mean_class_accuracy_epoch_30.pth'
