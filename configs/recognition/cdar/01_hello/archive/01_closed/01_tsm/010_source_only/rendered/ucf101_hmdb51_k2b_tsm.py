_base_ = [
    '../../../../_base_/data.py',
    '../../../../_base_/models/tsm_model.py',
    '../../../../../_base_/closed_tsm_training.py',
    '../../../../../../../_base_/default_runtime.py',
]

dataset_settings = dict(
    source=dict(
        train=dict(
            type='RawframeDataset',
            data_prefix='/local_datasets/ucf101/rawframes',
            ann_file='data/_filelists/ucf101/filelist_ucf_train_closed.txt'),
        valid=dict(
            type='RawframeDataset',
            data_prefix='/local_datasets/ucf101/rawframes',
            ann_file='data/_filelists/ucf101/filelist_ucf_val_closed.txt')),
    target=dict(
        test=dict(
            type='RawframeDataset',
            data_prefix='/local_datasets/hmdb51/rawframes',
            ann_file='data/_filelists/hmdb51/filelist_hmdb_val_closed.txt')),
)

data = dict(
    train=dict( **dataset_settings['source']['train']),
    val=dict(   **dataset_settings['source']['valid']),
    test=dict(  **dataset_settings['target']['test'])
)
