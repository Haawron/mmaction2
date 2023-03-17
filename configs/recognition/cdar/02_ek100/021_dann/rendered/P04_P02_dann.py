_base_ = [
    # model
    '../../_base_/tsf_warmup_training.py',
    # data
    '../../../../../_base_/default_runtime.py',
]

num_classes = 5
domain_adaptation = True
find_unused_parameters = True

# model settings
model = dict(
    type='DARecognizer3D',
    contrastive=False,
    backbone=dict(
        type='TimeSformer',
        pretrained=None,  # check load_from
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        frozen_stages=11,  # number of stages (total 12)
        norm_eval=False,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    neck=dict(
        type='DomainClassifier',
        in_channels=768,
        loss_weight=.5,
        num_layers=4,
        dropout_ratio=0.,
    ),
    cls_head=dict(type='TimeSformerHead', num_classes=num_classes, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score'))


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
            ann_file='data/_filelists/ek100/processed/filelist_P04_train_closed.txt'),
        test=dict(
            **datasets['RawframeDataset'],
            data_prefix=data_prefix,
            test_mode=True,
            ann_file='data/_filelists/ek100/processed/filelist_P04_train_closed.txt')),
    target=dict(
        train=dict(
            **datasets['RawframeDataset'],
            data_prefix=data_prefix,
            ann_file='data/_filelists/ek100/processed/filelist_P02_train_open_all.txt'),
        valid=dict(
            **datasets['RawframeDataset'],
            data_prefix=data_prefix,
            test_mode=True,
            ann_file='data/_filelists/ek100/processed/filelist_P02_test_merged_open_all.txt'),
        test=dict(
            **datasets['RawframeDataset'],
            data_prefix=data_prefix,
            test_mode=True,
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
    videos_per_gpu=48,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=32),
    train=[
        dict(
            **dataset_settings['source']['train'],
            pipeline=train_pipeline),
        dict(
            **dataset_settings['target']['train'],
            pipeline=train_pipeline),
    ],
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
