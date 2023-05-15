_base_ = [
    # data
    '../../../../__base__/tsm_training.py',
    # model
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
    UCF101=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
    ),
    HMDB51=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
    ),
)

# TODO: UCF-HMDB filelist 바꿔야 함!!!!!!!
dataset_settings = dict(
    source=dict(
        train=dict(
            **datasets['UCF101'],
            data_prefix='/local_datasets/ucf101/rawframes',
            ann_file='data/_filelists/ucf101/filelist_ucf_train_closed.txt'),
        valid=dict(
            **datasets['UCF101'],
            test_mode=True,
            data_prefix='/local_datasets/kinetics400/videos/val',
            ann_file='data/_filelists/ucf101/filelist_ucf_train_closed.txt')),
    target=dict(
        test=dict(
            **datasets['HMDB51'],
            test_mode=True,
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_test_closed.txt')))


img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
            dict(type='RawFrameDecode'),

            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='ColorJitter'),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        valid=[
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
