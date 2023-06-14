_base_ = [
    './k2b_i3d_lngn.py'
]

# model settings
model = dict(
    type='TemporallyPyramidicRecognizer',
    dim='3d',
    fuse_before_head=True,
    consensus_before_head=True,
    sampler_name='lngn',
    sampler_index=dict(l=[4, 5], g=3),
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
        _delete_=True,
        type='TimeSformerHead',
        in_channels=2048,
        num_classes=12,
    ),
)

datasets = dict(
    BABEL=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
    ),
    K400=dict(
        # type='VideoDataset',
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        start_index=0,  # denseflow로 푼 건 0부터 시작하고 opencv로 푼 건 1부터 시작함
    ),
)

dataset_settings = dict(
    source=dict(
        train=dict(
            **datasets['K400'],
            data_prefix='/local_datasets/kinetics400/rawframes_resized/train',
            ann_file='data/_filelists/k400/processed/filelist_k400_train_closed.txt'),
        valid=dict(
            **datasets['K400'],
            test_mode=True,
            data_prefix='/local_datasets/kinetics400/rawframes_resized/val',
            ann_file='data/_filelists/k400/processed/filelist_k400_val_closed.txt')),
    target=dict(
        test=dict(
            **datasets['BABEL'],
            test_mode=True,
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_test_closed.txt')))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

blend_options = dict(
    p=.25,  # 적용할 확률
    alpha=.75,  # alpha * 원본 클립 + (1 - alpha) * 배경
    resize_h=256, crop_size=224,
    ann_files=[
        'data/_filelists/k400/processed/filelist_k400_train_closed.txt',
    ],
    data_prefixes=[
        '/local_datasets/median/k400/train',
    ],
    blend_label=False
)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8),
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
        ],
    ),
)

data = dict(
    train=dict(
        **dataset_settings['source']['train'],
        pipeline=pipelines['source']['train'])
)
