_base_ = [
    '../rendered/ucf101_hmdb51_i3d.py'
]

# model settings
model = dict(
    backbone=dict(
        pretrained2d=False,
        pretrained=None,
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')),
    cls_head=dict(
        num_classes=12,
    ),
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root_source = '/local_datasets/ucf101/rawframes'
data_root_target = '/local_datasets/hmdb51/rawframes'
ann_file_train = 'data/_filelists/ucf101/processed/filelist_ucf_train_closed.txt'
ann_file_val = 'data/_filelists/ucf101/processed/filelist_ucf_val_closed.txt'
ann_file_test = 'data/_filelists/hmdb51/processed/filelist_hmdb_val_closed.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

blend_options = dict(
    p=.25,  # 적용할 확률
    alpha=.75,  # alpha * 원본 클립 + (1 - alpha) * 배경
    resize_h=256, crop_size=224,
    ann_files=[
        ann_file_train,
    ],
    data_prefixes=[
        '/local_datasets/median/ucf101/rawframes',
    ],
    blend_label=False
)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='BackgroundBlend', **blend_options),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    train=dict(pipeline=train_pipeline))
