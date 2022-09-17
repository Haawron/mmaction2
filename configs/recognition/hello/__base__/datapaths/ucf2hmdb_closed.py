# dataset settings
data_prefix_source = '/local_datasets/ucf101/rawframes'
data_prefix_target = '/local_datasets/hmdb51/rawframes'
ann_file_train_source = 'data/_filelists/ucf101/filelist_ucf_train_closed.txt'
ann_file_train_target = 'data/_filelists/hmdb51/filelist_hmdb_train_open.txt'
ann_file_valid_target = 'data/_filelists/hmdb51/filelist_hmdb_val_closed.txt'
ann_file_test_target = 'data/_filelists/hmdb51/filelist_hmdb_test_closed.txt'


def processor(data:dict):
    data['train'][0].update({'ann_file': ann_file_train_source, 'data_prefix': data_prefix_source})
    data['train'][1].update({'ann_file': ann_file_train_target, 'data_prefix': data_prefix_target})
    data['val'].update({'ann_file': ann_file_valid_target, 'data_prefix': data_prefix_target})
    data['test'].update({'ann_file': ann_file_test_target, 'data_prefix': data_prefix_target})
    return data
