# model settings
num_clips = 3
clip_len = 16
clip_interval = 8
frame_interval = 1

num_classes = 6  # =3!

domain_adaptation = True  # True to use DomainAdaptationRunner
find_unused_parameters = False  # True to resolve the issue when Freeze & dist


model = dict(
    type='DARecognizer2d',
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=clip_len,
        norm_eval=False,
        shift_div=8,),
        #frozen_stages=4),
    cls_head=dict(
        type='TSMCOPHead',
        loss_cls=dict(
            type='CrossEntropyLoss'),
        num_clips=num_clips,
        in_channels=2048,
        num_segments=clip_len,
        print_mca=True,

        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.1,
        init_std=0.001,
        is_shift=True),
    test_cfg=dict(average_clips='prob'))  # prob - prob, score - -distance, None - feature
# model training and testing settings
# dataset settings
data_prefix_source = '/local_datasets/ucf101/rawframes'
data_prefix_target = '/local_datasets/hmdb51/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/hmdb51/filelist_hmdb_train_open.txt'
ann_file_valid_target = 'data/_filelists/hmdb51/filelist_hmdb_val_closed.txt'
ann_file_test_target = 'data/_filelists/hmdb51/filelist_hmdb_test_closed.txt'
img_norm_cfg = dict(
    mean=[128., 128., 128.], std=[50., 50., 50.], to_bgr=False)

crop_size = 112
train_pipeline = [
    dict(
        type='COPSampleFrames',
        num_clips=num_clips,
        clip_len=clip_len,  # frames / clip
        clip_interval=clip_interval,
        frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1)),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='Resize', scale=(crop_size, crop_size), keep_ratio=False),
    # dict(type='Flip', flip_ratio=.5),
    # dict(type='ColorJitter', hue=.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='COPSampleFrames',
        num_clips=num_clips,
        clip_len=clip_len,  # frames / clip
        clip_interval=clip_interval,
        frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1)),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='COPSampleFrames',
        num_clips=num_clips,
        clip_len=clip_len,  # frames / clip
        clip_interval=clip_interval,
        frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1)),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=6,  # 여기가 gpu당 batch size임, source+target 한 번에 넣는 거라서 배치 사이즈 반절
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=20),
    train=[
        dict(
            type='COPRawframeDataset',  # Note this is not Contrastive- one
            num_clips=3,
            ann_file=ann_file_train_source,
            data_prefix=data_prefix_source,
            start_index=1,  # frame number starts with
            filename_tmpl='img_{:05}.jpg',
            sample_by_class=True,
            pipeline=train_pipeline),
        dict(
            type='COPRawframeDataset',
            num_clips=3,
            ann_file=ann_file_train_target,
            data_prefix=data_prefix_target,
            start_index=1,
            filename_tmpl='img_{:05}.jpg',
            sample_by_class=True,
            pipeline=train_pipeline),
    ],
    val=dict(
        type='COPRawframeDataset',
        num_clips=3,
        ann_file=ann_file_valid_target,
        data_prefix=data_prefix_target,
        start_index=1,
        filename_tmpl='img_{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type='COPRawframeDataset',
        num_clips=3,
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
)
total_epochs = 3000
checkpoint_config = dict(interval=300)
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
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth'
resume_from = None
workflow = [('train', 1)]
