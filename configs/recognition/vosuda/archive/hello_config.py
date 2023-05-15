# model settings
num_classes = 5+1  # 1: unknown


osbp_loss = dict(
    type='OSBPLoss',
    num_classes=num_classes,
    target_domain_label=.5)
model = dict(
    type='OSBPRecognizer2d',
    backbone=dict(
        # type='MobileNetV2TSM',
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='OSBPTSMHead',
        loss_cls=osbp_loss,
        num_classes=num_classes,
        num_segments=8,
        # in_channels=1280,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        # dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True))
# model training and testing settings
# train_cfg = None
# test_cfg = dict(average_clips='evidence', evidence_type='exp')
# dataset settings
data_root = 'data/epic-kitchens-100/EPIC-KITCHENS'
data_root_val = 'data/epic-kitchens-100/EPIC-KITCHENS'
ann_file_train_source = 'data/epic-kitchens-100/hello_filelist_02.txt'
ann_file_train_target = 'data/epic-kitchens-100/hello_filelist_22.txt'
ann_file_val = 'data/epic-kitchens-100/hello_filelist_22.txt'
ann_file_test = 'data/epic-kitchens-100/hello_filelist_22.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# clip_len vs. num_clips  https://github.com/open-mmlab/mmaction2/issues/1204
# clip_len: 한 clip 당 frame 수, defaults to 1
# num_clips: 한 비디오 당 sampling 할 clip 수
# 실수한 것
# 1. 보통 clip_len=1로 놓고 num_clips 수를 조절하나 봄
#   - 중구난방으로 몇 개 뽑아서 사용하는 것 같음.
#   - clip_len을 조절하면 top_k_accuracy 부분에서 에러가 남
# 2. 여기 mmaction에서의 video = 우리의 segment임
#   => mmaction에서의 clip = 우리의 segment의 일부분인 것임
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
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label', 'domain'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'domain'])
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
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,  # 여기가 gpu당 batch size임
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=2),
    train=dict(
        type='UDARawframeDataset',
        source_ann_file=ann_file_train_source,
        target_ann_file=ann_file_train_target,
        data_prefix=data_root,
        start_index=1,  # frame number starts with
        filename_tmpl='frame_{:010}.jpg',
        with_offset=True,
        pipeline=train_pipeline),
    val=dict(
        type='RawframeDataset',
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        start_index=1,
        filename_tmpl='frame_{:010}.jpg',
        with_offset=True,
        pipeline=val_pipeline),
    test=dict(
        type='RawframeDataset',
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        start_index=1,
        filename_tmpl='frame_{:010}.jpg',
        with_offset=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=1e-4,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
# lr_config = dict(policy='step', step=[20, 40])
lr_config = dict(
    policy='step', step=[40]
)
total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=2, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=5,  # every [ ] steps
    hooks=[
        dict(type='TextLoggerHook'),#, by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
annealing_runner = False
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/hello/OSBP_mobile'
load_from = None
resume_from = None
workflow = [('train', 1)]
