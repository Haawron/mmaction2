_base_ = [
    './_base_/k2b_gcd_tmf_data.py',
    '../_base_/k2b_training.py',
    # model
    '../../../../../_base_/default_runtime.py',
]

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
    neck=[
        dict(
            type='DomainClassifier',
            in_channels=768,
            loss_weight=.5,
            num_layers=4,
            dropout_ratio=0.),
        dict(
            type='Linear',
            in_channels=768,
            num_classes=num_classes,
            loss=dict(
                type='EvidenceLoss',
                num_classes=num_classes,
                evidence='exp',
                loss_type='log',
                with_kldiv=False,
                with_avuloss=True,
                annealing_method='exp')),
    ],
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
    test_cfg=dict(feature_extraction=True))
