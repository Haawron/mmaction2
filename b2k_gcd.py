datasets = dict(
    ContrastiveRawframeDataset=dict(
        type='ContrastiveRawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1),
    ContrastiveVideoDataset=dict(
        type='ContrastiveVideoDataset', sample_by_class=True),
    VideoDataset=dict(type='VideoDataset'))
dataset_settings = dict(
    source=dict(
        train=dict(
            type='ContrastiveRawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            with_offset=True,
            start_index=1,
            data_prefix='/local_datasets/babel',
            ann_file=
            'data/_filelists/babel/processed/filelist_babel_train_closed.txt'),
        test=dict(
            type='ContrastiveRawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            with_offset=True,
            start_index=1,
            test_mode=True,
            data_prefix='/local_datasets/babel',
            ann_file=
            'data/_filelists/babel/processed/filelist_babel_train_closed.txt')
    ),
    target=dict(
        train=dict(
            type='ContrastiveVideoDataset',
            sample_by_class=True,
            data_prefix='/local_datasets/kinetics400/videos/train',
            ann_file=
            'data/_filelists/k400/processed/filelist_k400_train_open_all.txt'),
        valid=dict(
            type='VideoDataset',
            test_mode=True,
            data_prefix='/local_datasets/kinetics400/videos/val',
            ann_file=
            'data/_filelists/k400/processed/filelist_k400_test_merged_open_all.txt'
        ),
        test=dict(
            type='VideoDataset',
            test_mode=True,
            data_prefix='/local_datasets/kinetics400/videos/train',
            ann_file=
            'data/_filelists/k400/processed/filelist_k400_train_open_all.txt'))
)
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
pipelines = dict(
    source=dict(
        train=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=32,
                num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='ColorJitter', hue=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        test=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=32,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    target=dict(
        train=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=32,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        valtest=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=32,
                num_clips=1,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]))
data = dict(
    videos_per_gpu=20,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=20),
    train=[
        dict(
            type='ContrastiveRawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            with_offset=True,
            start_index=1,
            data_prefix='/local_datasets/babel',
            ann_file=
            'data/_filelists/babel/processed/filelist_babel_train_closed.txt',
            pipeline=[
                dict(
                    type='SampleFrames',
                    clip_len=8,
                    frame_interval=32,
                    num_clips=1),
                dict(type='RawFrameDecode'),
                dict(type='RandomRescale', scale_range=(256, 320)),
                dict(type='RandomCrop', size=224),
                dict(type='Flip', flip_ratio=0.5),
                dict(type='ColorJitter', hue=0.5),
                dict(
                    type='Normalize',
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_bgr=False),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label'])
            ]),
        dict(
            type='ContrastiveVideoDataset',
            sample_by_class=True,
            data_prefix='/local_datasets/kinetics400/videos/train',
            ann_file=
            'data/_filelists/k400/processed/filelist_k400_train_open_all.txt',
            pipeline=[
                dict(type='DecordInit'),
                dict(
                    type='SampleFrames',
                    clip_len=8,
                    frame_interval=32,
                    num_clips=1),
                dict(type='DecordDecode'),
                dict(type='RandomRescale', scale_range=(256, 320)),
                dict(type='RandomCrop', size=224),
                dict(type='Flip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_bgr=False),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label'])
            ])
    ],
    val=dict(
        type='VideoDataset',
        test_mode=True,
        data_prefix='/local_datasets/kinetics400/videos/val',
        ann_file=
        'data/_filelists/k400/processed/filelist_k400_test_merged_open_all.txt',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=32,
                num_clips=1,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    test=dict(
        test_mode=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='ContrastiveRawframeDataset',
                filename_tmpl='img_{:05d}.jpg',
                with_offset=True,
                start_index=1,
                test_mode=True,
                data_prefix='/local_datasets/babel',
                ann_file=
                'data/_filelists/babel/processed/filelist_babel_train_closed.txt',
                pipeline=[
                    dict(
                        type='SampleFrames',
                        clip_len=8,
                        frame_interval=32,
                        num_clips=1,
                        test_mode=True),
                    dict(type='RawFrameDecode'),
                    dict(type='Resize', scale=(-1, 256)),
                    dict(type='CenterCrop', crop_size=224),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_bgr=False),
                    dict(type='FormatShape', input_format='NCTHW'),
                    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                    dict(type='ToTensor', keys=['imgs', 'label'])
                ]),
            dict(
                type='VideoDataset',
                test_mode=True,
                data_prefix='/local_datasets/kinetics400/videos/train',
                ann_file=
                'data/_filelists/k400/processed/filelist_k400_train_open_all.txt',
                pipeline=[
                    dict(type='DecordInit'),
                    dict(
                        type='SampleFrames',
                        clip_len=8,
                        frame_interval=32,
                        num_clips=1,
                        test_mode=True),
                    dict(type='DecordDecode'),
                    dict(type='Resize', scale=(-1, 256)),
                    dict(type='CenterCrop', crop_size=224),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_bgr=False),
                    dict(type='FormatShape', input_format='NCTHW'),
                    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                    dict(type='ToTensor', keys=['imgs', 'label'])
                ])
        ]))
evaluation = dict(
    interval=1,
    metrics=['kmeans', 'gcd_v2'],
    metric_options=dict(num_old_classes=12, num_all_classes=27),
    rule='greater',
    save_best='kmeans')
lr = 0.01
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        })),
    weight_decay=5e-05,
    nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=1e-05)
total_epochs = 3
work_dir = ''
load_from = None
num_classes = 12
domain_adaptation = True
find_unused_parameters = True
model = dict(
    type='DARecognizer3D',
    contrastive=True,
    backbone=dict(
        type='TimeSformer',
        pretrained='data/weights/vit/vit_base_patch16_224.pth',
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.0,
        transformer_layers=None,
        frozen_stages=11,
        norm_eval=False,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-06)),
    cls_head=dict(
        type='DINODAHead',
        in_channels=768,
        out_dim=65536,
        print_mca=False,
        loss_cls=dict(
            type='SemisupervisedContrastiveLoss',
            num_classes=12,
            unsupervised=True,
            loss_ratio=0.35,
            tau=1.0)),
    train_cfg=None,
    test_cfg=dict(feature_extraction=True))
checkpoint_config = None
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
