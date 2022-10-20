_base_ = [
    '../_base_/gcd4da_svt_model.py',
    '../_base_/gcd4da_svt_training.py',
    '../../../../_base_/default_runtime.py',
]

num_classes = 12


# dataset settings
data_prefix_source = '/local_datasets/ucf101/rawframes'
data_prefix_target = '/local_datasets/hmdb51/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/hmdb51/filelist_hmdb_train_open.txt'
ann_file_valid_target = 'data/_filelists/hmdb51/filelist_hmdb_val_open.txt'
ann_file_test_target = 'data/_filelists/hmdb51/filelist_hmdb_test_open.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(
        type='BackgroundBlend',
        p=.5,
        resize_h=256, crop_size=224,
        ann_files=[ann_file_train_source, ann_file_train_target],
        data_prefixes=[
            '/local_datasets/median/ucf101/rawframes',
            '/local_datasets/median/hmdb51/rawframes'],
        alpha='random',  # blend ratio of origin image
        blend_label=False),
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
data = dict(
    videos_per_gpu=20,  # 여기가 gpu당 batch size임, source+target 한 번에 넣는 거라서 배치 사이즈 반절
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
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
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
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_test_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
        ])
)

# load_from = 'work_dirs/train_output/ucf2hmdb/svt/vanilla/source-only/4370__vanilla-svt-ucf2hmdb-source-only/2/20220728-182744/best_mean_class_accuracy_epoch_5.pth'
