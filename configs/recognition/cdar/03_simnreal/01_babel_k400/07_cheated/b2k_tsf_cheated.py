_base_ = [
    './_base_/b2k_tsf_cheated_data.py',
    # '../_base_/b2k_training.py',
    # '../../_base_/tsf_warmup_model.py',
    '../../../../../_base_/default_runtime.py',
]


######################################################################################

evaluation = dict(
    interval=10,
    metrics=['gcd_v2_cheated', 'confusion_matrix', 'gcd_v2', 'kmeans'],  # valid, test 공용으로 사용, 아 이러면 마지막 metric의 confmat만 보여주네
    metric_options={'num_old_classes': 12, 'num_all_classes': 27},
    rule='greater',
    save_best='kmeans')

# optimizer
lr=1e-2
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
total_epochs = 50
work_dir = './work_dirs/train_output/hello/cdar/gcd'
load_from = None

######################################################################################

num_classes = 27
domain_adaptation = False
find_unused_parameters = False

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=None,  # check load_from
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        frozen_stages=-1,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=num_classes, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score'))
