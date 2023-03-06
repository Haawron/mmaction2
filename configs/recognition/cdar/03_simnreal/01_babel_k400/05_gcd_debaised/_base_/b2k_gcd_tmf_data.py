datasets = dict(
    ContrastiveRawframeDataset=dict(
        type='ContrastiveRawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
    ),
    RawframeDataset=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
    ),
    ContrastiveVideoDataset=dict(
        type='ContrastiveVideoDataset',
        sample_by_class=True,
    ),
    VideoDataset=dict(
        type='VideoDataset',
    ),
)

# isn't this a duplication of `data`? => No
# the difference is the hierarchy
# this:   domain -> split
# `data`: split -> domain
dataset_settings = dict(
    source=dict(
        train=dict(
            **datasets['ContrastiveRawframeDataset'],
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_train_closed.txt'),
        test=dict(
            **datasets['RawframeDataset'],
            test_mode=True,
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_train_closed.txt')),
    target=dict(
        train=dict(
            **datasets['ContrastiveVideoDataset'],
            data_prefix='/local_datasets/kinetics400/videos/train',
            ann_file='data/_filelists/k400/processed/filelist_k400_train_open_all.txt'),
        valid=dict(
            **datasets['VideoDataset'],
            test_mode=True,
            data_prefix='/local_datasets/kinetics400/videos/val',
            ann_file='data/_filelists/k400/processed/filelist_k400_test_merged_open_all.txt'),
        test=dict(
            **datasets['VideoDataset'],
            test_mode=True,
            data_prefix='/local_datasets/kinetics400/videos/train',
            ann_file='data/_filelists/k400/processed/filelist_k400_train_open_all.txt')))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

blend_options = dict(
    p=.5, resize_h=256, crop_size=224,
    ann_files=[
        # dataset_settings['source']['train']['ann_file'],
        dataset_settings['target']['train']['ann_file']],
    data_prefixes=[
        # '/local_datasets/median/babel/train',
        '/local_datasets/median/k400/train'],
    alpha='random',  # blend ratio of origin image
    blend_label=False
)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomCrop', size=224),
            dict(type='BackgroundBlend', **blend_options),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='ColorJitter', hue=.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        test=[
            dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
    target=dict(
        train=[
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
            dict(type='DecordDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomCrop', size=224),
            dict(type='BackgroundBlend', **blend_options),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        valtest=[
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1, test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
)
data = dict(
    videos_per_gpu=20,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=20),
    train=[
        dict(
            **dataset_settings['source']['train'],
            pipeline=pipelines['source']['train']),
        dict(
            **dataset_settings['target']['train'],
            pipeline=pipelines['target']['train']),
    ],
    val=dict(  # k-means
        **dataset_settings['target']['valid'],
        pipeline=pipelines['target']['valtest']),
    test=dict(  # SS k-means
        test_mode=True,
        type='ConcatDataset',
        datasets=[
            dict(
                **dataset_settings['source']['test'],
                pipeline=pipelines['source']['test']),
            dict(
                **dataset_settings['target']['test'],
                pipeline=pipelines['target']['valtest']),
        ]
    )
)
