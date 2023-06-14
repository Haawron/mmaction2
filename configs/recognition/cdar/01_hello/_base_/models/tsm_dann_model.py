num_classes = 12
domain_adaptation = True
find_unused_parameters = False

# model settings
model = dict(
    type='DARecognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    neck=dict(
        type='DomainClassifier',
        in_channels=2048,
        loss_weight=1,
        num_layers=4,
        dropout_ratio=.5,
        backbone='TSM', num_segments=8,
    ),
    cls_head=dict(
        type='TSMHead',
        num_classes=num_classes,
        in_channels=2048,
        dropout_ratio=0.5),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score'))
