_base_ = [
    # './_base_/vanilla_svt_model_ek100_closed.py',
    './_base_/vanilla_svt_data_P02_closed.py',
    './_base_/vanilla_svt_training_closed.py',
    '../../../_base_/default_runtime.py',
]

num_classes = 5
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
    cls_head=dict(type='TimeSformerHead', num_classes=num_classes, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob')) 
