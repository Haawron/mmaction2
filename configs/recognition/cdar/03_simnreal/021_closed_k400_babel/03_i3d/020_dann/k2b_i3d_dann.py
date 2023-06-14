_base_ = [
    '../../../../../../_base_/models/i3d_r50.py',
    '../../../../../../_base_/schedules/sgd_50e.py',
    '../../../../../../_base_/default_runtime.py'
]

domain_adaptation = True
find_unused_parameters = False

# model settings
model = dict(
    type='DARecognizer3D',
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
        type='DomainClassifier',
        in_channels=2048,
        loss_weight=1,
        num_layers=4,
        dropout_ratio=.5,
        backbone='TimeSformer',
    ),
    cls_head=dict(
        spatial_type=None,
        num_classes=12,
    ),
)

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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
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
    ),
    target=dict(
        train=[
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
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
            dict(type='FormatShape', input_format='TCNHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        valid=[
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='TCNHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        test=[
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=32, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='TCNHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
    ),
)
data = dict(
    videos_per_gpu=18,
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=40),
    test_dataloader=dict(videos_per_gpu=8),
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
        pipeline=pipelines['target']['valid']),
    test=dict(
        **dataset_settings['target']['test'],
        pipeline=pipelines['target']['test'])
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
    paramwise_cfg=dict(fc_lr5=True),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4)
optimizer_config = dict(
    # type='GradientCumulativeOptimizerHook',  # 지금은 노필요
    # cumulative_iters=2,
    grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[8, 16])
total_epochs = 20
work_dir = './work_dirs/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/'
load_from = 'data/weights/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth'
