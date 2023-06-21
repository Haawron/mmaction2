_base_ = [
    '../../../../../../_base_/models/i3d_r50.py',
    '../../../../../../_base_/schedules/sgd_50e.py',
    '../../../../../../_base_/default_runtime.py'
]

domain_adaptation = True


model = dict(
    type='DARecognizer3D',
    samplings=dict(source='dense', target='dense'),
    backbone=dict(
        pretrained2d=False,
        pretrained=None,
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')),
    neck=dict(
        type='VCOPN',
        in_channels=2048,
        num_clips=3,
        backbone='TimeSformer'),
    cls_head=dict(
        type='IdentityHead'),
)

# dataset settings
dataset_type = 'RawframeDataset'
data_prefix = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
ann_file_train_source = 'data/_filelists/ek100/mm-sada_resampled/D1_train.txt'
ann_file_train_target = 'data/_filelists/ek100/mm-sada_resampled/D3_train.txt'

dataset_settings = dict(
    type=dataset_type,
    filename_tmpl='frame_{:010}.jpg',
    with_offset=True,
    start_index=1,
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=3),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),

            dict(type='Resize', scale=(224, 224), keep_ratio=False),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
    target=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=3),
            dict(type='RawFrameDecode'),

            dict(type='Resize', scale=(224, 224), keep_ratio=False),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
    ),
)

data = dict(
    videos_per_gpu=18,
    workers_per_gpu=6,
    train=[
        dict(
            **dataset_settings,
            ann_file=ann_file_train_source,
            data_prefix=data_prefix,
            pipeline=pipelines['source']['train']),
        dict(
            **dataset_settings,
            ann_file=ann_file_train_target,
            data_prefix=data_prefix,
            pipeline=pipelines['target']['train']),
    ])

# runtime settings
log_config = dict(interval=10)
checkpoint_config = dict(interval=25)
total_epochs = 200
lr_config = dict(policy='step', step=[int(total_epochs*.95)])
work_dir = './work_dirs/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/'
load_from = 'data/weights/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth'
