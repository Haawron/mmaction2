
num_classes = 12
domain_adaptation = True
find_unused_parameters = True

# model settings
model = dict(
    type='DARecognizer3D',
    contrastive=True,  # if True, don't shuffle the chosen batch
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
        frozen_stages=11,
        norm_eval=False,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='DINODAHead',
        in_channels=768,
        out_dim=65536,
        loss_cls=dict(
            type='SemisupervisedContrastiveLoss',
            num_classes=num_classes,  # gonna be $k$ (for phase1)
            unsupervised=True,
            loss_ratio=.35,
            tau=1.)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(feature_extraction=True))  # None: prob - prob, score - -distance, None - feature

# dataset settings
data_prefix_source = '/local_datasets/ucf101/rawframes'
data_prefix_target = '/local_datasets/hmdb51/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/hmdb51/filelist_hmdb_train_open.txt'
ann_file_valid_target = 'data/_filelists/hmdb51/filelist_hmdb_val_open.txt'
ann_file_test_target = 'data/_filelists/hmdb51/filelist_hmdb_test_open.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', hue=.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=20,  # (frozen, B) = (10, 12), (11, 20)
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
            sample_by_class=False,
            pipeline=train_pipeline),
    ],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_source,
                data_prefix=data_prefix_source,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=val_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_target,
                data_prefix=data_prefix_target,
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
                ann_file=ann_file_train_source,
                data_prefix=data_prefix_source,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_train_target,
                data_prefix=data_prefix_target,
                start_index=1,
                num_classes=num_classes,
                filename_tmpl='img_{:05}.jpg',
                pipeline=test_pipeline),
            dict(
                type='RawframeDataset',
                ann_file=ann_file_valid_target,
                data_prefix=data_prefix_target,
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

evaluation = dict(
    interval=5,
    metrics=['sskmeans'],  # valid, test 공용으로 사용
    metric_options=dict(sskmeans=dict(fixed_k=22, n_tries=100)),
    # ckpt-saving options
    save_best='sskmeans', rule='greater')

# optimizer
lr=.1
optimizer = dict(
    type='SGD',
    lr=lr,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=5e-5,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=lr*1e-3)
total_epochs = 200

# runtime settings
checkpoint_config = dict(interval=10)

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
work_dir = './work_dirs/hello/ucf2hmdb/svt/gcd4da/'
load_from = 'data/weights/svt/releases/download/v1.0/SVT_mmaction.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
