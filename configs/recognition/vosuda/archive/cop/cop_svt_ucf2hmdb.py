# model settings
num_clips = 3
# clip_len = 16
clip_len = 8
clip_interval = 8
frame_interval = 1

num_classes = 6  # =3!

domain_adaptation = True  # True to use DomainAdaptationRunner
find_unused_parameters = True

model = dict(
    type='DARecognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=None,  # check load_from
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        frozen_stages=-1,
        norm_eval=False,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='TimeSFormerCOPHead',
        num_clips=num_clips,
        num_hidden=512,
        in_channels=768),
    test_cfg=dict(average_clips='prob'))
# model training and testing settings
# dataset settings
data_prefix_source = '/local_datasets/ucf101/rawframes'
data_prefix_target = '/local_datasets/hmdb51/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/hmdb51/filelist_hmdb_train_open.txt'
ann_file_valid_target = 'data/_filelists/hmdb51/filelist_hmdb_val_closed.txt'
ann_file_test_target = 'data/_filelists/hmdb51/filelist_hmdb_test_closed.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

resize_size = 256
crop_size = 224  # ViT라서 크기 맞춰줘야 pre-trained weight 쓸 수 있음
train_pipeline = [
    dict(
        type='COPSampleFrames',
        num_clips=num_clips,
        clip_len=clip_len,  # frames / clip
        clip_interval=clip_interval,
        frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(resize_size, -1)),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='Resize', scale=(crop_size, crop_size), keep_ratio=False),
    # dict(type='Flip', flip_ratio=.5),
    # dict(type='ColorJitter', hue=.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
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
    dict(type='Resize', scale=(resize_size, -1)),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='Resize', scale=(crop_size, crop_size), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='COPSampleFrames',
        num_clips=num_clips,
        clip_len=clip_len,  # frames / clip
        clip_interval=clip_interval,
        frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(resize_size, -1)),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='Resize', scale=(crop_size, crop_size), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=16,  # 여기가 gpu당 batch size임, source+target 한 번에 넣는 거라서 배치 사이즈 반절
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
    lr=0.005,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step', step=[20, 40]
)
total_epochs = 1000
checkpoint_config = dict(interval=200)
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],  # valid, test 공용으로 사용
    metric_options=dict(
        top_k_accuracy=dict(topk=(1, 5)),
        use_predefined_labels=True),
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
work_dir = './work_dirs/hello/ucf2hmdb/svt/cop/'
load_from = 'data/weights/svt/releases/download/v1.0/SVT_mmaction.pth'
resume_from = None
workflow = [('train', 1)]
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
