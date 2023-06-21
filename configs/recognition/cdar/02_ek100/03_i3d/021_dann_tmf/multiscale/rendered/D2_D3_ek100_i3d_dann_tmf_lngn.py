_base_ = [
    '../../../../../../../_base_/models/i3d_r50.py',
    '../../../../../../../_base_/schedules/sgd_50e.py',
    '../../../../../../../_base_/default_runtime.py'
]

domain_adaptation = True
find_unused_parameters = False

domain_classifier_common_options = dict(
    num_layers=4,
    dropout_ratio=.5,
    backbone='TimeSformer'
)

# model settings
model = dict(
    type='TemporallyPyramidicDARecognizer',
    dim='3d',
    sampler_name='lngn',
    sampler_index=dict(l=[4, 5], g=3),
    backbone=dict(
        pretrained2d=False,
        pretrained=None,
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')),
    neck=[
        dict(
            type='TemporallyPyramidicDomainClassifier',
            temporal_locality='local',  # local, global, both
            fusion_method='mean',  # '', concat, mean
            loss_weight=1.,
            in_channels=2048,
            hidden_dim=2048,
            nickname='l',
            **domain_classifier_common_options),
        dict(
            type='TemporallyPyramidicDomainClassifier',
            temporal_locality='global',
            fusion_method='mean',
            loss_weight=1,
            in_channels=2048,
            hidden_dim=2048,
            nickname='g',
            **domain_classifier_common_options),
        dict(
            type='TemporallyPyramidicDomainClassifier',
            temporal_locality='cross',
            loss_weight=1.,
            in_channels=2048,
            hidden_dim=2048,
            nickname='x',
            **domain_classifier_common_options),
    ],
    cls_head=dict(
        spatial_type=None,
        num_classes=8,
    ),
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
ann_file_train_source = 'data/_filelists/ek100/mm-sada_resampled/D2_train.txt'
ann_file_train_target = 'data/_filelists/ek100/mm-sada_resampled/D3_train.txt'
ann_file_val   = 'data/_filelists/ek100/mm-sada_resampled/D3_test.txt'
ann_file_test  = 'data/_filelists/ek100/mm-sada_resampled/D3_test.txt'

dataset_settings = dict(
    type=dataset_type,
    filename_tmpl='frame_{:010}.jpg',
    with_offset=True,
    start_index=1,
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

blend_options = dict(
    p=.25,  # 적용할 확률
    alpha=.75,  # alpha * 원본 클립 + (1 - alpha) * 배경
    resize_h=256, crop_size=224,
    ann_files=[],  # all imgs included in each data_prefix
    data_prefixes=[
        '/local_datasets/median/ek100-mm-sada-split/train/D2',
        '/local_datasets/median/ek100-mm-sada-split/train/D3',
    ],
    blend_label=False
)

pipelines = dict(
    source=dict(
        train=[
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
        ],
    ),
    target=dict(
        train=[
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

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        valid=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        test=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
    ),
)

data = dict(
    videos_per_gpu=12,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=4),  # 왜 valid 때 터지는지 모르겠음
    train=[
        dict(
            **dataset_settings,
            ann_file=ann_file_train_source,
            data_prefix=data_root,
            pipeline=pipelines['source']['train']),
        dict(
            **dataset_settings,
            ann_file=ann_file_train_target,
            data_prefix=data_root,
            pipeline=pipelines['target']['train']),
    ],
    val=dict(
        **dataset_settings,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=pipelines['target']['valid']),
    test=dict(
        **dataset_settings,
        ann_file=ann_file_test,
        data_prefix=data_root,
        pipeline=pipelines['target']['test'])
)

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],
    save_best='top1_acc')

# runtime settings
log_config = dict(interval=20)
checkpoint_config = dict(interval=10)
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50

work_dir = './work_dirs/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/'
load_from = ''
