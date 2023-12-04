_base_ = [
    '../../../../../../../_base_/models/i3d_r50.py',
    '../../../../../../../_base_/schedules/sgd_50e.py',
    '../../../../../../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='TemporallyPyramidicRecognizer',
    dim='3d',
    fuse_before_head=True,
    consensus_before_head=True,
    sampler_name='lngn',
    sampler_index=dict(l=[3, 4, 5]),
    backbone=dict(
        pretrained2d=False,
        pretrained=None,
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')),
    neck=dict(
        type='VCOPN4GLA',
        in_channels=2048,
        num_clips=3,
        dropout_ratio=.0,
        backbone='TimeSformer'),
    cls_head=dict(
        _delete_=True,
        type='TimeSformerHead',
        in_channels=2048,
        num_classes=12,
    ),
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root_source = '/local_datasets/hmdb51/rawframes'
data_root_target = '/local_datasets/ucf101/rawframes'
ann_file_train = 'data/_filelists/hmdb51/processed/filelist_hmdb_train_closed.txt'
ann_file_val = 'data/_filelists/hmdb51/processed/filelist_hmdb_val_closed.txt'
ann_file_test = 'data/_filelists/ucf101/processed/filelist_ucf_val_closed.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

blend_options = dict(
    p=.25,  # 적용할 확률
    alpha=.75,  # alpha * 원본 클립 + (1 - alpha) * 배경
    resize_h=256, crop_size=224,
    ann_files=[
        ann_file_train,
    ],
    data_prefixes=[
        '/local_datasets/median/hmdb51/rawframes',
    ],
    blend_label=False
)

train_pipeline=[
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8),
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
    dict(type='Flip', flip_ratio=0.5),
    dict(type='BackgroundBlend', **blend_options),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline=[
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline=[
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]


data = dict(
    videos_per_gpu=24,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        start_index=1,
        ann_file=ann_file_train,
        data_prefix=data_root_source,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        start_index=1,
        ann_file=ann_file_val,
        data_prefix=data_root_source,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        start_index=1,
        ann_file=ann_file_test,
        data_prefix=data_root_target,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],
    save_best='mean_class_accuracy')

# runtime settings
log_config = dict(interval=3)
checkpoint_config = dict(interval=25)
total_epochs = 50
work_dir = './work_dirs/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/'
load_from = ''