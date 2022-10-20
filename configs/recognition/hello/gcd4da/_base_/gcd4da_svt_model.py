num_classes = 12
domain_adaptation = True
find_unused_parameters = True

# model settings
model = dict(
    type='DARecognizer3D',
    contrastive=True,  # if True, don't shuffle the chosen batch
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
        frozen_stages=11,
        norm_eval=False,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='DINODAHead',
        in_channels=768,
        out_dim=65536,
        print_mca=False,
        loss_cls=dict(
            type='SemisupervisedContrastiveLoss',
            num_classes=num_classes,  # gonna be $k$ (for phase1)
            unsupervised=True,
            loss_ratio=.35,
            tau=1.)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(feature_extraction=True))  # None: prob - prob, score - -distance, None - feature
