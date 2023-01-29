# dataset settings
dataset_settings = dict(
    source=dict(
        dataset_type='RawframeDataset',
        train=dict(
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_train_closed.txt'),
        test=dict(
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_train_closed.txt')),
    target=dict(
        dataset_type='VideoDataset',
        valid=dict(
            data_prefix='/local_datasets/kinetics400/videos/val',
            ann_file='data/_filelists/k400/processed/filelist_k400_test_merged_open_all.txt'),
        test=dict(
            data_prefix='/local_datasets/kinetics400/videos/train',
            ann_file='data/_filelists/k400/processed/filelist_k400_train_open_all.txt')))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    # Colorjitter?
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

valtest_pipeline_target = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=10,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=20),
    train=dict(
        type=dataset_settings['source']['dataset_type'],
        ann_file=dataset_settings['source']['train']['ann_file'],
        data_prefix=dataset_settings['source']['train']['data_prefix'],
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
        pipeline=train_pipeline),
    val=dict(
        test_mode=True,
        type=dataset_settings['target']['dataset_type'],
        ann_file=dataset_settings['target']['valid']['ann_file'],
        data_prefix=dataset_settings['target']['valid']['data_prefix'],
        pipeline=valtest_pipeline_target),
    test=dict(
        test_mode=True,
        type='ConcatDataset',
        datasets=[
            dict(
                test_mode=True,
                type=dataset_settings['source']['dataset_type'],
                ann_file=dataset_settings['source']['test']['ann_file'],
                data_prefix=dataset_settings['source']['test']['data_prefix'],
                filename_tmpl='img_{:05d}.jpg',
                with_offset=True,
                start_index=1,
                pipeline=test_pipeline),
            dict(
                test_mode=True,
                type=dataset_settings['target']['dataset_type'],
                ann_file=dataset_settings['target']['test']['ann_file'],
                data_prefix=dataset_settings['target']['test']['data_prefix'],
                pipeline=valtest_pipeline_target),
        ]
    )
)
