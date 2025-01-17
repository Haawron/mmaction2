_base_ = ['./__base__/osbp_tsm_ucfhmdb.py']

from configs.recognition.hello.osbp.__base__.osbp_tsm_ucfhmdb import img_norm_cfg
from configs.recognition.hello.osbp.osbp_tsm_hmdb2ucf import data, ann_file_train_source, ann_file_train_target


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
    dict(
        type='BackgroundBlend',
        p=.5,
        resize_h=256, crop_size=224,
        ann_files=[ann_file_train_source, ann_file_train_target],
        data_prefixes=[
            '/local_datasets/median/hmdb51/rawframes',
            '/local_datasets/median/ucf101/rawframes'],
        alpha='random',  # blend ratio of origin image
        blend_label=False),
    dict(type='Flip', flip_ratio=.5),
    dict(type='ColorJitter', hue=.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data['train'][0]['pipeline'] = train_pipeline
data['train'][1]['pipeline'] = train_pipeline
load_from = 'work_dirs/train_output/hmdb2ucf/tsm/vanilla/source-only/4382__vanilla-tsm-hmdb2ucf-source-only/2/20220728-205928/best_mean_class_accuracy_epoch_30.pth'
