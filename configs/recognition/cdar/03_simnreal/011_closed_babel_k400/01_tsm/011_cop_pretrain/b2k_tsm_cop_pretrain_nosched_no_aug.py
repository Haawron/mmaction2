_base_ = [
    # '../03_gcd/_base_/k2b_gcd_data.py',
    # '../__base__/closed_k2b_tsm_training.py',
    # '../../_base_/tsf_warmup_model.py',
    '../../../../../../_base_/default_runtime.py',
]


# model settings
num_classes = 12

domain_adaptation = True

model = dict(
    type='DARecognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    neck=dict(
        type='VCOPN',
        in_channels=2048,
        num_clips=3,
        backbone='TSM', num_segments=8),
    cls_head=dict(
        type='IdentityHead'),
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
        type='VideoDataset',
    ),
)

dataset_settings = dict(
    source=dict(
        train=dict(
            **datasets['BABEL'],
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_train_closed.txt')),
    target=dict(
        train=dict(
            **datasets['K400'],
            data_prefix='/local_datasets/kinetics400/videos/train',
            ann_file='data/_filelists/k400/processed/filelist_k400_train_closed.txt')))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=3),
            dict(type='RawFrameDecode'),

            dict(type='Resize', scale=(224, 224), keep_ratio=False),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
    target=dict(
        train=[
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=3),
            dict(type='DecordDecode'),

            dict(type='Resize', scale=(224, 224), keep_ratio=False),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
)
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=12,
    val_dataloader=dict(videos_per_gpu=40),
    train=[
        dict(
            **dataset_settings['source']['train'],
            pipeline=pipelines['source']['train']),
        dict(
            **dataset_settings['target']['train'],
            pipeline=pipelines['target']['train']),
    ]
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
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=48,
    grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
total_epochs = 500
lr_config = dict(policy='step', step=[int(total_epochs*.8)])
work_dir = './work_dirs/train_output/hello/cdar/tsm'
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb_20230317-be0fc26e.pth'
ckpt_revise_keys = []#[('cls_head', 'unusedhead')]