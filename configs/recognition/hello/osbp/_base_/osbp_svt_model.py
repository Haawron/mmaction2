num_classes = 12+1
domain_adaptation = True
find_unused_parameters = True

# model settings
model = dict(
    type='DARecognizer3D',
    contrastive=False,
    backbone=dict(
        type='TimeSformer',
        pretrained=None,  # check load_from
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        frozen_stages=11,  # number of stages (total 12)
        norm_eval=False,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='OSBPDINODAHead',
        loss_cls=dict(
            type='OSBPLoss',
            num_classes=num_classes,
            target_domain_label=.7,
            weight_loss=1e-3),
        in_channels=768,
        num_classes=num_classes),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score'))
