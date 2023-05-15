# model settings
num_classes = 12

domain_adaptation = True

model = dict(
    type='DARecognizer2D',
    contrastive=True,  # if True, don't shuffle the chosen batch
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='ContrastiveDATSMHead',
        loss_cls=dict(
            type='SemisupervisedContrastiveLoss',
            num_classes=num_classes,  # gonna be $k$
            unsupervised=True,
            loss_ratio=.35,
            tau=1.),
        num_classes=num_classes,
        in_channels=2048,
        num_layers=1,
        num_features=512,
        num_segments=8,

        # 이거 그대로 써도 되겠다
        # hsic=0이면 그냥 contrastive만 하는 거잖슴
        debias=False,
        bias_input=False,
        bias_network=False,
        debias_last=True,
        hsic_factor=0,

        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.1,
        init_std=0.001,
        is_shift=True),
    test_cfg=dict(average_clips='prob'))  # None: prob - prob, score - -distance, None - feature
# model training and testing settings
# dataset settings
data_prefix_source = '/local_datasets/hmdb51/rawframes'
data_prefix_target = '/local_datasets/ucf101/rawframes'
ann_file_train_source = 'data/_filelists/hmdb51/filelist_hmdb_train_closed.txt'
ann_file_train_target = 'data/_filelists/ucf101/filelist_ucf_train_open_all.txt'
ann_file_valid_target = 'data/_filelists/ucf101/filelist_ucf_val_closed.txt'
ann_file_test_target = 'data/_filelists/ucf101/filelist_ucf_test_closed.txt'
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
    videos_per_gpu=4,  # 여기가 gpu당 batch size임, source+target 한 번에 넣는 거라서 배치 사이즈 반절
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
        type='RawframeDataset',
        ann_file=ann_file_valid_target,
        data_prefix=data_prefix_target,
        start_index=1,
        filename_tmpl='img_{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type='RawframeDataset',
        ann_file=ann_file_test_target,
        data_prefix=data_prefix_target,
        start_index=1,
        filename_tmpl='img_{:05}.jpg',
        pipeline=test_pipeline)
)
# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    lr=4 * 1e-3,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(
    policy='step', step=[20, 40],
    # warmup
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1,  # start from [ratio * base_lr]
)
total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],  # valid, test 공용으로 사용
    save_best='mean_class_accuracy')
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
work_dir = 'work_dirs/hello/ucf2hmdb/tsm/cdar'
load_from = 'work_dirs/train_output/hmdb2ucf/tsm/vanilla/source-only/3514__tsm-hmdb-vanilla-source-only/1/20220722-023128/best_mean_class_accuracy_epoch_20.pth'
resume_from = None
workflow = [('train', 1)]
