# model settings
num_classes = 12

domain_adaptation = True
find_unused_parameters = True

model = dict(
    type='DARecognizer2D',
    contrastive=True,  # if True, don't shuffle the chosen batch
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        frozen_stages=3,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMDINODAHead',
        in_channels=2048,
        out_dim=65536,
        loss_cls=dict(
            type='SemisupervisedContrastiveLoss',
            num_classes=num_classes,  # gonna be $k$ (for phase1)
            unsupervised=True,
            loss_ratio=.35,
            tau=1.)),
    test_cfg=dict(feature_extraction=True))  # None: prob - prob, score - -distance, None - feature

# dataset settings
data_prefix_source = '/local_datasets/ucf101/rawframes'
data_prefix_target = '/local_datasets/hmdb51/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/hmdb51/filelist_hmdb_train_open.txt'
ann_file_valid_source = 'data/_filelists/ucf101/filelist_ucf_val_closed.txt'
ann_file_valid_target = 'data/_filelists/hmdb51/filelist_hmdb_val_open.txt'
ann_file_test_source = 'data/_filelists/ucf101/filelist_ucf_test_closed.txt'
ann_file_test_target = 'data/_filelists/hmdb51/filelist_hmdb_test_open.txt'
img_norm_cfg = dict(
    mean=[128., 128., 128.], std=[50., 50., 50.], to_bgr=False)
train_pipeline = [
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
    dict(type='Flip', flip_ratio=.5),
    dict(type='ColorJitter', hue=.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=28,  # 여기가 gpu당 batch size임, source+target 한 번에 넣는 거라서 배치 사이즈 반절
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=20),
    train=[
        dict(
            type='ContrastiveRawframeDataset',
            ann_file=ann_file_train_source,
            data_prefix=data_prefix_source,
            start_index=1,  # frame number starts with
            filename_tmpl='img_{:05}.jpg',
            sample_by_class=True,
            pipeline=train_pipeline),
        dict(
            type='ContrastiveRawframeDataset',
            ann_file=ann_file_train_target,
            data_prefix=data_prefix_target,
            start_index=1,
            filename_tmpl='img_{:05}.jpg',
            sample_by_class=True,
            pipeline=train_pipeline),
    ],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_source,
                data_prefix=data_prefix_source,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
        ]),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RawframeDataset',
                ann_file=ann_file_test_source,
                data_prefix=data_prefix_source,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_test_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
        ])
)
lr=.1
# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    lr=lr,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=lr*1e-3)
total_epochs = 200
checkpoint_config = dict(interval=10)
work_dir = 'work_dirs/hello/ucf2hmdb/tsm/gcd4da'
load_from = 'work_dirs/train_output/ucf2hmdb/tsm/vanilla/source-only/4380__vanilla-tsm-ucf2hmdb-source-only/4/20220728-204923/best_mean_class_accuracy_epoch_20.pth'
evaluation = dict(
    interval=5,
    metrics=['sskmeans'],  # valid, test 공용으로 사용
    metric_options=dict(sskmeans=dict(fixed_k=22, n_tries=100)),
    # ckpt-saving options
    save_best='sskmeans', rule='greater')
log_config = dict(
    interval=10,  # every [ ] steps
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
annealing_runner = False
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
