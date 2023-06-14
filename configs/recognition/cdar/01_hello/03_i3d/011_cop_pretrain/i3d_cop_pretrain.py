_base_ = [
    '../010_source_only/rendered/ucf_hmdb_i3d.py'
]

domain_adaptation = True


model = dict(
    type='DARecognizer3D',
    samplings=dict(source='dense', target='dense'),
    neck=dict(
        type='VCOPN',
        in_channels=2048,
        num_clips=3,
        backbone='TimeSformer'),
    cls_head=dict(
        type='IdentityHead'),
)

dataset_settings = dict(
    source=dict(
        train=dict(
            type='RawframeDataset',
            start_index=1,
            data_prefix='/local_datasets/ucf101/rawframes',
            ann_file='data/_filelists/ucf101/processed/filelist_ucf_train_closed.txt')),
    target=dict(
        train=dict(
            type='RawframeDataset',
            start_index=1,
            data_prefix='/local_datasets/hmdb51/rawframes',
            ann_file='data/_filelists/hmdb51/processed/filelist_hmdb_train_closed.txt')))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=3),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),

            dict(type='Resize', scale=(224, 224), keep_ratio=False),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
    target=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=3),
            dict(type='RawFrameDecode'),

            dict(type='Resize', scale=(224, 224), keep_ratio=False),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
    ),
)

data = dict(
    videos_per_gpu=18,
    workers_per_gpu=8,
    train=[
        dict(
            **dataset_settings['source']['train'],
            pipeline=pipelines['source']['train']),
        dict(
            **dataset_settings['target']['train'],
            pipeline=pipelines['target']['train']),
    ])

total_epochs = 100
lr_config = dict(policy='step', step=[int(total_epochs*.95)])
