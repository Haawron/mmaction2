evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'H_mean_class_accuracy', 'mean_class_accuracy', 'confusion_matrix'],  # valid, test 공용으로 사용
    save_best='H_mean_class_accuracy')

# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=4 * 1e-5,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
