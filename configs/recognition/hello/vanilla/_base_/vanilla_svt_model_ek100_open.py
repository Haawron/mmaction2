num_classes = 6
domain_adaptation = False
find_unused_parameters = True

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=None,  # check load_from
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        frozen_stages=11,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='DINOHead',
        in_channels=768,
        num_classes=num_classes,
        print_mca=False),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob')) 

