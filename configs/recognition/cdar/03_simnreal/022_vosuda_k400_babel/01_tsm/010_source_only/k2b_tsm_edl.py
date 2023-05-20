_base_ = [
    # '../03_gcd/_base_/k2b_gcd_data.py',
    '../__base__/k2b_tsm_training.py',
    # '../../_base_/tsf_warmup_model.py',
    '../../../../../../_base_/default_runtime.py',
]


# model settings
num_classes = 12

domain_adaptation = False

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained=None,
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMHead',
        num_classes=num_classes,
        loss_cls=dict(
            type='EvidenceLoss',
            num_classes=num_classes),
        num_segments=8,
        in_channels=2048,
        dropout_ratio=0.5,
        openset=True),
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
        # type='VideoDataset',
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        start_index=0,  # denseflow로 푼 건 0부터 시작하고 opencv로 푼 건 1부터 시작함
    ),
)

dataset_settings = dict(
    source=dict(
        train=dict(
            **datasets['K400'],
            data_prefix='/local_datasets/kinetics400/rawframes_resized/train',
            ann_file='data/_filelists/k400/processed/filelist_k400_train_closed.txt'),
        valid=dict(
            **datasets['K400'],
            test_mode=True,
            data_prefix='/local_datasets/kinetics400/rawframes_resized/val',
            ann_file='data/_filelists/k400/processed/filelist_k400_val_open.txt')),
    target=dict(
        valid=dict(
            **datasets['BABEL'],
            test_mode=True,
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_val_open.txt'),
        test=dict(
            **datasets['BABEL'],
            test_mode=True,
            data_prefix='/local_datasets/babel',
            ann_file='data/_filelists/babel/processed/filelist_babel_test_open.txt')))


img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

pipelines = dict(
    source=dict(
        train=[
            dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),

            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        valid=[
            dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
    target=dict(
        valtest=[
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
    ),
)

data = dict(
    videos_per_gpu=24,
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=40),
    train=dict(
        **dataset_settings['source']['train'],
        pipeline=pipelines['source']['train']),
    val=dict(
        **dataset_settings['target']['valid'],
        pipeline=pipelines['target']['valtest']),
    test=dict(
        **dataset_settings['target']['test'],
        pipeline=pipelines['target']['valtest'])
)

#####################################################################################################################
# training

# evaluation = dict(
#     interval=5,
#     metrics=['top_k_accuracy', 'mean_class_accuracy', 'H_mean_class_accuracy', 'confusion_matrix'],  # valid, test 공용으로 사용
#     save_best='mean_class_accuracy')
