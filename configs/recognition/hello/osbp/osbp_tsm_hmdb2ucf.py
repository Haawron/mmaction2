from configs.recognition.hello.osbp.__base__.osbp_tsm_ucfhmdb import data, evaluation, work_dir
_base_ = ['./__base__/osbp_tsm_ucfhmdb.py']


# dataset settings
data_prefix_source = '/local_datasets/hmdb51/rawframes'
data_prefix_target = '/local_datasets/ucf101/rawframes'
ann_file_train_source = 'data/_filelists/hmdb51/filelist_hmdb_train_closed.txt'
ann_file_train_target = 'data/_filelists/ucf101/filelist_ucf_train_open.txt'
ann_file_valid_target = 'data/_filelists/ucf101/filelist_ucf_val_open.txt'
ann_file_test_target = 'data/_filelists/ucf101/filelist_ucf_test_open.txt'

data['train'][0].update({'ann_file': ann_file_train_source, 'data_prefix': data_prefix_source})
data['train'][1].update({'ann_file': ann_file_train_target, 'data_prefix': data_prefix_target})
data['val'].update({'ann_file': ann_file_valid_target, 'data_prefix': data_prefix_target})
data['test'].update({'ann_file': ann_file_test_target, 'data_prefix': data_prefix_target})

load_from = 'work_dirs/train_output/hmdb2ucf/tsm/vanilla/source-only/4382__vanilla-tsm-hmdb2ucf-source-only/2/20220728-205928/best_mean_class_accuracy_epoch_30.pth'
evaluation['metrics'].append('logits')
evaluation['metric_options'] = dict(logits=dict(p_out_dir=work_dir))
