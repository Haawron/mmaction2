_base_ = [
    '../../../../../../_base_/models/i3d_r50.py',
    '../../../../../../_base_/schedules/sgd_50e.py',
    '../../../../../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        pretrained2d=False,
        pretrained=None,
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')),
    cls_head=dict(
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
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=24,
    workers_per_gpu=8,
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
checkpoint_config = dict(interval=10)
total_epochs = 20
work_dir = './work_dirs/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/'
load_from = 'data/weights/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth'
