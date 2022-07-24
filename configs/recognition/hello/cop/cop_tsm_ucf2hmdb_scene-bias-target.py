# model settings
num_classes = 365  # Places365

domain_adaptation = False  # True to use DomainAdaptationRunner
find_unused_parameters = True  # True to resolve the issue when Freeze & dist


model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8,
        frozen_stages=4),
    cls_head=dict(
        type='DANNTSMHead',
        loss_cls=dict(
            type='DANNClassifierLoss',
            num_classes=num_classes),
        loss_domain=dict(
            type='DANNDomainLoss',
            loss_weight=0),  # to eval scene bias
        num_classes=num_classes,
        num_cls_layers=1,
        num_domain_layers=4,
        num_segments=8,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    test_cfg=dict(average_clips='score'))
# model training and testing settings
# dataset settings
data_prefix = '/local_datasets/hmdb51/rawframes'
ann_file_train = 'data/_filelists/hmdb51/scene/filelist_scene_hmdb_train_open.txt'
ann_file_valid = 'data/_filelists/hmdb51/scene/filelist_scene_hmdb_val_closed.txt'
ann_file_test = 'data/_filelists/hmdb51/scene/filelist_scene_hmdb_test_closed.txt'
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
    videos_per_gpu=16,  # 여기가 gpu당 batch size임, source+target 한 번에 넣는 거라서 배치 사이즈 반절
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=20),
    train=dict(
        type='RawframeDataset',
        ann_file=ann_file_train,
        data_prefix=data_prefix,
        start_index=1,  # frame number starts with
        filename_tmpl='img_{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type='RawframeDataset',
        ann_file=ann_file_valid,
        data_prefix=data_prefix,
        start_index=1,
        filename_tmpl='img_{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type='RawframeDataset',
        ann_file=ann_file_test,
        data_prefix=data_prefix,
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
)
total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],  # valid, test 공용으로 사용
    save_best='mean_class_accuracy')
log_config = dict(
    interval=30,  # every [ ] steps
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
annealing_runner = False
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/hello/ucf2hmdb/tsm/cop'
load_from = 'work_dirs/train_output/ucf2hmdb/tsm/cop_dann/3211__tsm-cop-ucf2hmdb-dann/0/20220719-211401/best_mean_class_accuracy_epoch_40.pth'
resume_from = None
workflow = [('train', 1)]
