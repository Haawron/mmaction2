_base_ = [
    '../_base_/gcd4da_svt_model.py',
    '../_base_/gcd4da_svt_training.py',
    # '../_base_/gcd4da_svt_data_hmdb2ucf.py',
    '../../../../_base_/default_runtime.py',
]

_num_classes = 12  # '_' at start: to avoid conflicting keys among bases

# dataset settings
data_prefix_source = '/local_datasets/hmdb51/rawframes'
data_prefix_target = '/local_datasets/ucf101/rawframes'
ann_file_train_source = 'data/_filelists/hmdb51/filelist_hmdb_train_closed.txt'
ann_file_train_target = 'data/_filelists/ucf101/filelist_ucf_train_open.txt'
ann_file_valid_target = 'data/_filelists/ucf101/filelist_ucf_val_open.txt'
ann_file_test_target = 'data/_filelists/ucf101/filelist_ucf_test_open.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Imgaug', transforms='default'),
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
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=20,  # (frozen, B) = (10, 12), (11, 20)
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=20),
    train=[
        dict(
            type='ContrastiveRawframeDataset',
            ann_file=ann_file_train_source,
            data_prefix=data_prefix_source,
            start_index=1,  # frame number starts with
            filename_tmpl='img_{:05}.jpg',
            sample_by_class=True,
            pipeline=train_pipeline),
        dict(
            type='ContrastiveRawframeDataset',
            ann_file=ann_file_train_target,
            data_prefix=data_prefix_target,
            start_index=1,
            filename_tmpl='img_{:05}.jpg',
            sample_by_class=True,
            pipeline=train_pipeline),
    ],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_source,
                data_prefix=data_prefix_source,
                start_index=1,
                num_classes=_num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=_num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=_num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=_num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
        ]),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_source,
                data_prefix=data_prefix_source,
                start_index=1,
                num_classes=_num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=_num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=_num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_test_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=_num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
        ])
)
