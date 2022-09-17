from configs.recognition.hello.gcd4da.plain.__base__.gcd4da_phase0_tsm_ucfhmdb import data, evaluation
_base_ = ['./__base__/gcd4da_phase0_tsm_ucfhmdb.py']


# dataset settings
data_prefix_source = '/local_datasets/ucf101/rawframes'
data_prefix_target = '/local_datasets/hmdb51/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/hmdb51/filelist_hmdb_train_open.txt'
ann_file_valid_target = 'data/_filelists/hmdb51/filelist_hmdb_val_open.txt'
ann_file_test_target = 'data/_filelists/hmdb51/filelist_hmdb_test_open.txt'

data['train'][0].update({'ann_file': ann_file_train_source, 'data_prefix': data_prefix_source})
data['train'][1].update({'ann_file': ann_file_train_target, 'data_prefix': data_prefix_target})
data['val'].update({'ann_file': ann_file_valid_target, 'data_prefix': data_prefix_target})
data['test'].update({'ann_file': ann_file_test_target, 'data_prefix': data_prefix_target})

work_dir = 'work_dirs/hello/ucf2hmdb/tsm/gcd4da'
load_from = 'work_dirs/train_output/ucf2hmdb/tsm/vanilla/source-only/4380__vanilla-tsm-ucf2hmdb-source-only/4/20220728-204923/best_mean_class_accuracy_epoch_20.pth'

evaluation['metrics'] += ['logits'] if 'logits' not in evaluation['metrics'] else []
evaluation['metric_options'] = dict(logits=dict(p_out_dir=work_dir))
