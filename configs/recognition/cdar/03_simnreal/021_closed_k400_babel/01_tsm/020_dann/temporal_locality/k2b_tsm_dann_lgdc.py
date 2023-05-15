_base_ = [
    # '../03_gcd/_base_/k2b_gcd_data.py',
    # '../../__base__/closed_k2b_tsm_training.py',
    # '../../_base_/tsf_warmup_model.py',
    '../../../../../../../_base_/default_runtime.py',
]


#####################################################################################################################
# model

num_classes = 12
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
    sampler_name='lngn',
    sampler_index=dict(l=[3, 4], g=3),
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    neck=[
        dict(
            type='TemporallyPyramidicDomainClassifier',
            temporal_locality='local-global',
            loss_weight=1.,
            in_channels=2048,
            hidden_dim=2048,
            nickname='lg',
            **domain_classifier_common_options),
    ],
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=num_classes,
        in_channels=2048),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score'))

#####################################################################################################################
# data

datasets = dict(
    BABEL=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
    ),
    K400=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        start_index=0,
    ),
)

dataset_settings = dict(
    source=dict(
        train=dict(
            **datasets['K400'],
            data_prefix='/local_datasets/kinetics400/rawframes_resized/train',
            ann_file='data/_filelists/k400/processed/filelist_k400_train_closed.txt'),
        test=dict(
            **datasets['K400'],
            test_mode=True,
            data_prefix='/local_datasets/kinetics400/rawframes_resized/val',
            ann_file='data/_filelists/k400/processed/filelist_k400_test_closed.txt')),
    target=dict(
        train=dict(
            **datasets['BABEL'],
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_train_closed.txt'),
        valid=dict(
            **datasets['BABEL'],
            test_mode=True,
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_val_closed.txt'),
        test=dict(
            **datasets['BABEL'],
            test_mode=True,
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_test_closed.txt')))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

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
            # dict(type='ColorJitter'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
    ),
    target=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        valtest=[
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
)
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=10,
    val_dataloader=dict(videos_per_gpu=4),
    train=[
        dict(
            **dataset_settings['source']['train'],
            pipeline=pipelines['source']['train']),
        dict(
            **dataset_settings['target']['train'],
            pipeline=pipelines['target']['train']),
    ],
    val=dict(
        **dataset_settings['target']['valid'],
        pipeline=pipelines['target']['valtest']),
    test=dict(
        **dataset_settings['target']['test'],
        pipeline=pipelines['target']['valtest'])
)

#####################################################################################################################
# training

evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],  # valid, test 공용으로 사용
    save_best='mean_class_accuracy')

# optimizer
lr=1e-3
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',  # 지금은 노필요
    cumulative_iters=3,
    grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 20
work_dir = './work_dirs/train_output/hello/cdar/tsm'
load_from = ''
