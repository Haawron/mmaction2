# model settings
num_classes = 5+1  # 1: unknown


osbp_loss_dict = dict(
    type='OSBPLoss',
    num_classes=num_classes,
    target_domain_label=.5,
    weighting_loss=True,
)
model = dict(
    type='OSBPRecognizer2d',
    backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='OSBPTSMHead',
        loss_cls=osbp_loss_dict,
        num_classes=num_classes,
        num_segments=8,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    test_cfg=dict(average_clips='prob'))
# model training and testing settings
# dataset settings
# data_root = 'data/epic-kitchens-100/EPIC-KITCHENS'
# data_root_val = 'data/epic-kitchens-100/EPIC-KITCHENS'
data_root = '/local_datasets/epic-kitchens-100/EPIC-KITCHENS'
data_root_val = '/local_datasets/epic-kitchens-100/EPIC-KITCHENS'
ann_file_train_source = 'data/epic-kitchens-100/filelist_P02_train_closed.txt'
ann_file_train_target = 'data/epic-kitchens-100/filelist_P22_train_open.txt'
ann_file_valid_source = 'data/epic-kitchens-100/filelist_P02_valid_closed.txt'  
ann_file_valid_target = 'data/epic-kitchens-100/empty.txt'  # 실제론 training 중에 target accuracy를 얻을 수 없으니 빈 파일로 두는 게 맞음
ann_file_test_source = 'data/epic-kitchens-100/empty.txt'  # open-set detection accuracy 잴 때만 사용
ann_file_test_target = 'data/epic-kitchens-100/filelist_P22_test_open.txt'  # 실제 accuracy + open-set detection
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label', 'domain'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'domain'])
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
    dict(type='Collect', keys=['imgs', 'label', 'domain'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'domain'])
]
data = dict(
    videos_per_gpu=24,  # 여기가 gpu당 batch size임
    workers_per_gpu=2,
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
        type='UDARawframeDataset',
        source_ann_file=ann_file_valid_source,
        target_ann_file=ann_file_valid_target,  # empty file이지만 mmaction 내부에서 loss 게산할 때 써야 돼서 이렇게 workaround로 구현함
        data_prefix=data_root_val,
        start_index=1,
        filename_tmpl='frame_{:010}.jpg',
        with_offset=True,
        pipeline=val_pipeline),
    test=dict(
        type='UDARawframeDataset',
        source_ann_file=ann_file_test_source,
        target_ann_file=ann_file_test_target,
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
    lr=4 * 1e-5,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(
    policy='step', step=[20, 40]
)
total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=10, metrics=['top_k_accuracy', 'mean_class_accuracy'])  # valid, test 공용으로 사용
log_config = dict(
    interval=10,  # every [ ] steps
    hooks=[
        dict(type='TextLoggerHook'),#, by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
annealing_runner = False
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/OSBP_02_to_22'
load_from = None
resume_from = None
workflow = [('train', 1)]