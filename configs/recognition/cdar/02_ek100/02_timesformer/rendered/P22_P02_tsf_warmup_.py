_base_ = [
    '../_base_/tsf_warmup_model.py',
    '../_base_/tsf_warmup_training.py',
    # data
    '../../../../_base_/default_runtime.py',
]
# dataset settings
datasets = dict(
    RawframeDataset=dict(
        type='RawframeDataset',
        filename_tmpl='frame_{:010}.jpg',
        with_offset=True,
        start_index=1,
    ),
)

data_prefix = '/local_datasets/epic-kitchens-100/EPIC-KITCHENS'

dataset_settings = dict(
    source=dict(
        train=dict(
            **datasets['RawframeDataset'],
            data_prefix=data_prefix,
            ann_file='data/_filelists/ek100/processed/filelist_P22_train_closed.txt'),
        test=dict(
            **datasets['RawframeDataset'],
            test_mode=True,
            data_prefix=data_prefix,
            ann_file='data/_filelists/ek100/processed/filelist_P22_train_closed.txt')),
    target=dict(
        valid=dict(
            **datasets['RawframeDataset'],
            test_mode=True,
            data_prefix=data_prefix,
            ann_file='data/_filelists/ek100/processed/filelist_P02_test_merged_open_all.txt'),
        test=dict(
            **datasets['RawframeDataset'],
            test_mode=True,
            data_prefix=data_prefix,
            ann_file='data/_filelists/ek100/processed/filelist_P02_train_open_all.txt')))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
valtest_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
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
        **dataset_settings['source']['train'],
        pipeline=train_pipeline),
    val=dict(  # k-means
        **dataset_settings['target']['valid'],
        pipeline=valtest_pipeline),
    test=dict(  # SS k-means
        test_mode=True,
        type='ConcatDataset',
        datasets=[
            dict(
                **dataset_settings['source']['test'],
                pipeline=valtest_pipeline),
            dict(
                **dataset_settings['target']['test'],
                pipeline=valtest_pipeline),
        ]
    )
)
