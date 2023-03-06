from slurm.gcd4da.commons.kmeans import SSKMeansTrainer as KMeans
from mmaction.datasets.custom_metrics import split_cluster_acc_v2, split_cluster_acc_v2_balanced
import numpy as np
from pathlib import Path
import re
import os
import csv
import inspect
from itertools import permutations


tasks = {
    'hello': ['ucf', 'hmdb'],
    'ek100': ['P02', 'P04', 'P22'],
    'simnreal': ['k400', 'babel']
}
real_dataset_name_mapping = {
    'ucf': ['ucf101', 'ucf'],
    'hmdb': ['hmdb51', 'hmdb'],
    'P02': ['ek100', 'P02'],
    'P04': ['ek100', 'P04'],
    'P22': ['ek100', 'P22'],
    'k400': ['k400', 'k400'],
    'babel': ['babel', 'babel'],
}
num_classes_mapping = {
    'ucf': [12, 22],
    'hmdb': [12, 22],
    'P02': [5, 15],
    'P04': [5, 15],
    'P22': [5, 15],
    'k400': [12, 27],
    'babel': [12, 20],
}

END = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[92m'
PURPLE = '\033[95m'


def get_ann_from_domainname(domain:str):
    dataset_dirname, dataset_filelist_name = real_dataset_name_mapping[domain]
    p_ann_dir = p_filelist_base / dataset_dirname
    if domain not in ['ucf', 'hmdb']:
        p_ann_dir /= 'processed'
    p_ann = p_ann_dir / f'filelist_{dataset_filelist_name}_train_open_all.txt'
    assert p_ann.is_file()
    with p_ann.open() as f:
        ann = np.array([int(line[-1]) for line in csv.reader(f, delimiter=' ')])
    return ann


def get_features_from_domainname(task:str, domain:str):
    weight = 'k400' if task != 'simnreal' else 'in1k'
    p_pkl = Path(f'/data/hyogun/repos/haawron_mmaction2/data/features/{weight}/{task}/{domain}/train.pkl')
    pkl = np.array(np.load(p_pkl, allow_pickle=True))
    return pkl


def print_conf(conf, num_old_classes=None):
    h, w = conf.shape
    num_old_classes = num_old_classes or h
    with np.printoptions(threshold=np.inf, linewidth=np.inf):  # thres: # elems, width: # chars
        s = str(conf)
    if 'SRUN_DEBUG' in os.environ:  # if in srun(debugging) session
        for (ii, jj), (start, end) in reversed([((i//w, i%h), (m.start(0), m.end(0))) for i, m in enumerate(re.finditer(r'\d+', str(s))) if i%h == i//w]):
            s = s[:start] + BOLD + (GREEN if max(ii, jj) < num_old_classes else PURPLE) + s[start:end] + END + s[end:]  # diag vals to be bold
    print('\nConfmat (gt/pred)\n' + s + '\n')


print('\n\nLoading features and annotations...')
models = {}
p_filelist_base = Path(f"/data/hyogun/repos/haawron_mmaction2/data/_filelists")
for task in tasks:
    for source, target in permutations(tasks[task], 2):
        if task == 'ek100':
            target, source = source, target
        ann_source = get_ann_from_domainname(source)
        ann_target = get_ann_from_domainname(target)
        pkl_source = get_features_from_domainname(task, source)
        pkl_target = get_features_from_domainname(task, target)
        assert ann_source.shape[0] == pkl_source.shape[0]
        assert ann_target.shape[0] == pkl_target.shape[0]
        num_old_classes, num_all_classes = num_classes_mapping[target]
        print(source, target)
        print('\t', ann_source.shape,  ann_target.shape)
        print('\t', pkl_source.shape,  pkl_target.shape)

        # just building
        kmeans = KMeans(autoinit=False, num_known_classes=num_old_classes, ks=num_all_classes)
        kmeans.Xs = {
            'train_source': pkl_source[ann_source<num_old_classes],
            'train_target': pkl_target,
            'valid': pkl_target,
            'test': pkl_target,
        }
        kmeans.anns = {
            'train_source': ann_source[ann_source<num_old_classes],
            'train_target': ann_target,
            'valid': ann_target,
            'test': ann_target,
        }
        models[f'{source}_{target}'] = kmeans

print('\n\nTraining SS k-means models...')
for task in tasks:
    for source, target in permutations(tasks[task], 2):
        if task == 'ek100':
            target, source = source, target
        num_old_classes, num_all_classes = num_classes_mapping[target]
        print(source, target)
        kmeans = models[f'{source}_{target}']
        kmeans.train()

        ann_target = kmeans.anns['test']
        pred_target = kmeans.predict(kmeans.model_best)['test']
        old_mask = (ann_target < num_old_classes)
        total_acc, old_acc, new_acc, conf = split_cluster_acc_v2(ann_target, pred_target, old_mask, True)
        log = inspect.cleandoc(f'''
        GCD V2 
            ALL: {100*total_acc:4.1f}
            Old: {100*old_acc:4.1f}
            New: {100*new_acc:4.1f}
        ''') + '\n\n'
        total_acc, old_acc, new_acc = split_cluster_acc_v2_balanced(ann_target, pred_target, old_mask)
        log += inspect.cleandoc(f'''
        GCD V2 Balanced
            ALL: {100*total_acc:4.1f}
            Old: {100*old_acc:4.1f}
            New: {100*new_acc:4.1f}
        ''') + '\n'
        print(log)
        print_conf(conf, num_old_classes)
        print('\n' + '='*200 + '\n')
