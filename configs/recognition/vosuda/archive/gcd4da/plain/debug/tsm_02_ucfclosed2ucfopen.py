from configs.recognition.hello.gcd4da.plain.__base__.gcd4da_phase0_tsm_ucfhmdb import data, evaluation, num_classes
_base_ = ['../__base__/gcd4da_phase0_tsm_ucfhmdb_runtime.py']


find_unused_parameters = True

model = dict(
    type='DARecognizer2D',
    contrastive=True,  # if True, don't shuffle the chosen batch
    backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        frozen_stages=3,
        depth=50,
        num_segments=8,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMDINODAHead',
        in_channels=2048,
        out_dim=65536,
        loss_cls=dict(
            type='SemisupervisedContrastiveLoss',
            num_classes=num_classes,  # gonna be $k$ (for phase1)
            unsupervised=True,
            loss_ratio=.35)),
    test_cfg=dict(average_clips='prob'))  # None: prob - prob, score - -distance, None - feature
# dataset settings
data_prefix_source = '/local_datasets/ucf101/rawframes'
data_prefix_target = '/local_datasets/ucf101/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/ucf101/filelist_ucf_train_open.txt'
ann_file_valid_target = 'data/_filelists/ucf101/filelist_ucf_val_closed.txt'
ann_file_test_target = 'data/_filelists/ucf101/filelist_ucf_test_closed.txt'

data['videos_per_gpu'] = 28
data['train'][0].update({'ann_file': ann_file_train_source, 'data_prefix': data_prefix_source})
data['train'][1].update({'ann_file': ann_file_train_target, 'data_prefix': data_prefix_target})
data['val'].update({'ann_file': ann_file_valid_target, 'data_prefix': data_prefix_target})
data['test'].update({'ann_file': ann_file_test_target, 'data_prefix': data_prefix_target})

lr=5e-5
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    lr=lr,
    momentum=0.9,
    weight_decay=0.0001)
lr_config = dict(policy='CosineAnnealing', min_lr=lr*1e-3)

total_epochs = 200
work_dir = 'work_dirs/hello/ucf2hmdb/tsm/gcd4da'
load_from = 'work_dirs/train_output/ucf2hmdb/tsm/vanilla/source-only/4380__vanilla-tsm-ucf2hmdb-source-only/4/20220728-204923/best_mean_class_accuracy_epoch_20.pth'

evaluation['metrics'] += ['logits'] if 'logits' not in evaluation['metrics'] else []
evaluation['metric_options'] = dict(logits=dict(p_out_dir=work_dir))
