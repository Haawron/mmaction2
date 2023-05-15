_base_ = [
    # '../03_gcd/_base_/k2b_gcd_data.py',
    '../__base__/closed_k2b_tsm_training.py',
    # '../../_base_/tsf_warmup_model.py',
    '../../../../../../_base_/default_runtime.py',
]


# model settings
num_classes = 12

domain_adaptation = False

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMHead',
        num_classes=num_classes,
        num_segments=8,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    test_cfg=dict(average_clips='score'))

#####################################################################################################################
# data

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
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

blend_options = dict(
    p=.25,  # 적용할 확률
    alpha=.75,  # alpha * 원본 클립 + (1 - alpha) * 배경
    resize_h=256, crop_size=224,
    ann_files=[
        dataset_settings['source']['train']['ann_file'],
        # dataset_settings['target']['train']['ann_file']
    ],
    data_prefixes=[
        '/local_datasets/median/k400/train',
        # '/local_datasets/median/babel/train',
    ],
    blend_label=False
)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
            dict(type='RawFrameDecode'),

            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='BackgroundBlend', **blend_options),
            dict(type='ColorJitter'),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        valid=[
            dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
    target=dict(
        test=[
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
)

data = dict(
    videos_per_gpu=24,
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=40),
    train=dict(
        **dataset_settings['source']['train'],
        pipeline=pipelines['source']['train']),
    val=dict(
        **dataset_settings['source']['valid'],
        pipeline=pipelines['source']['valid']),
    test=dict(
        **dataset_settings['target']['test'],
        pipeline=pipelines['target']['test'])
)
