# model settings
num_classes = 12

domain_adaptation = True
find_unused_parameters = True

model = dict(
    type='DARecognizer2D',
    contrastive=True,  # if True, don't shuffle the chosen batch
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        frozen_stages=3,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMDINODAHead',
        in_channels=2048,
        out_dim=65536,
        loss_cls=dict(
            type='SemisupervisedContrastiveLoss',
            num_classes=num_classes,  # gonna be $k$ (for phase1)
            unsupervised=True,
            loss_ratio=.35,
            tau=1.)),
    test_cfg=dict(feature_extraction=True))  # None: prob - prob, score - -distance, None - feature
