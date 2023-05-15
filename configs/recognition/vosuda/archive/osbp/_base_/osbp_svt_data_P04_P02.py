# dataset settings
data_prefix = '/local_datasets/epic-kitchens-100/EPIC-KITCHENS'
ann_file_train_source = 'data/_filelists/ek100/filelist_P04_train_closed.txt'
ann_file_train_target = 'data/_filelists/ek100/filelist_P02_train_open.txt'
ann_file_valid_target = 'data/_filelists/ek100/filelist_P02_valid_open.txt'
ann_file_test_target = 'data/_filelists/ek100/filelist_P02_test_open.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', hue=.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
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
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

filename_tmpl = 'frame_{:010}.jpg'
data = dict(
    videos_per_gpu=44,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=20),
    train=[
        dict(
            type='RawframeDataset',
            ann_file=ann_file_train_source,
            data_prefix=data_prefix,
            start_index=1,  # frame number starts with
            filename_tmpl=filename_tmpl,
            with_offset=True,
            sample_by_class=True,
            pipeline=train_pipeline),
        dict(
            type='RawframeDataset',
            ann_file=ann_file_train_target,
            data_prefix=data_prefix,
            start_index=1,
            filename_tmpl=filename_tmpl,
            with_offset=True,
            pipeline=train_pipeline),
    ],
    val=dict(
        type='RawframeDataset',
        ann_file=ann_file_valid_target,
        data_prefix=data_prefix,
        start_index=1,
        filename_tmpl=filename_tmpl,
        with_offset=True,
        pipeline=val_pipeline),
    test=dict(
        type='RawframeDataset',
        ann_file=ann_file_test_target,
        data_prefix=data_prefix,
        start_index=1,
        filename_tmpl=filename_tmpl,
        with_offset=True,
        pipeline=test_pipeline)
)
