_base_ = [
    './k2b_i3d.py'
]

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/local_datasets/kinetics400/rawframes_resized/train'
data_root_val = '/local_datasets/kinetics400/rawframes_resized/val'
data_root_test = '/local_datasets/babel'
ann_file_train = 'data/_filelists/k400/processed/filelist_k400_train_closed.txt'
ann_file_val = 'data/_filelists/k400/processed/filelist_k400_val_closed.txt'
ann_file_test = 'data/_filelists/babel/processed/filelist_babel_test_closed.txt'

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
        '/local_datasets/median/k400/train',
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
