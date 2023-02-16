_base_ = [
    '../_base_/default_runtime.py',
]

batch_size = 4

# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='PlacesHead',
        batch_size=batch_size,
        num_classes=365,
        in_channels=2048,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score'))

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # to_bgr=False 해야 되나?

pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=30, frame_interval=1, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=10,
    workers_per_gpu=24,
    test_dataloader=dict(videos_per_gpu=batch_size),
    test=dict(
        test_mode=True,
        type='VideoDataset',
        ann_file='data/_filelists/k400/processed/filelist_k400_train_open_all.txt',
        data_prefix='/local_datasets/kinetics400/videos/train',
        pipeline=pipeline
    )
)
