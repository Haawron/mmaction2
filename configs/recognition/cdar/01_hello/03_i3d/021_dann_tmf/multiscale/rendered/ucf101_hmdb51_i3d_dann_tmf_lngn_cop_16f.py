_base_ = [
    '../../../../../../../_base_/models/i3d_r50.py',
    '../../../../../../../_base_/schedules/sgd_50e.py',
    '../../../../../../../_base_/default_runtime.py'
]

domain_adaptation = True
find_unused_parameters = False

domain_classifier_common_options = dict(
    num_layers=4,
    dropout_ratio=.1,
    backbone='TimeSformer'
)

# model settings
num_locals = 3
model = dict(
    type='TemporallyPyramidicDARecognizer',
    dim='3d',
    sampler_name='just_split',
    sampler_index=dict(l=num_locals, g=1),  # not an index, nomenclature is misled. tech debt.
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
            type='VCOPN4GLA',
            in_channels=2048,
            num_clips=num_locals,
            loss_weight=1.,
            dropout_ratio=0.,
            backbone='TimeSformer'),
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
        num_classes=12,
    ),
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root_source = '/local_datasets/ucf101/rawframes'
data_root_target = '/local_datasets/hmdb51/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/processed/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/hmdb51/processed/filelist_hmdb_train_closed.txt'
ann_file_val          = 'data/_filelists/hmdb51/processed/filelist_hmdb_val_closed.txt'
ann_file_test         = 'data/_filelists/hmdb51/processed/filelist_hmdb_val_closed.txt'

dataset_settings = dict(
    type=dataset_type,
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
        '/local_datasets/median/ucf101',
        '/local_datasets/median/hmdb51'
    ],
    blend_label=False
)

sampler_base = dict(
    type='MultiScaleSampleFrames',
    num_locals=num_locals,
    num_globals=1,
    # options for both
    clip_len=16,
    # options for local
    frame_interval_local=2,
    # options for global
    dense_sampling=False,
    #frame_interval_global=8,
)

samplers = dict(
    train=sampler_base,
    valid=dict(**sampler_base, test_mode=True),
    test=dict(
        type='MultiScaleSampleFrames',
        num_locals=8,
        num_globals=2,
        # options for both
        clip_len=16,
        # options for local
        frame_interval_local=2,
        # options for global
        dense_sampling=False,
        #frame_interval_global=8,
        test_mode=True,
    ),
)
pipelines = dict(
    source_train=[
        samplers['train'],
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
    target_train=[
        samplers['train'],
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
        samplers['valid'],
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ],
    test=[
        samplers['test'],
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ],
)

data = dict(
    videos_per_gpu=6,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=[
        dict(
            **dataset_settings,
            ann_file=ann_file_train_source,
            data_prefix=data_root_source,
            pipeline=pipelines['source_train']),
        dict(
            **dataset_settings,
            ann_file=ann_file_train_target,
            data_prefix=data_root_target,
            pipeline=pipelines['target_train']),
    ],
    val=dict(
        **dataset_settings,
        ann_file=ann_file_val,
        data_prefix=data_root_target,
        pipeline=pipelines['valid']),
    test=dict(
        **dataset_settings,
        ann_file=ann_file_test,
        data_prefix=data_root_target,
        pipeline=pipelines['test'])
)

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'confusion_matrix'],
    save_best='top1_acc')

# runtime settings
log_config = dict(interval=10)
checkpoint_config = dict(interval=25)
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50

work_dir = './work_dirs/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/'
load_from = ''
