import csv
from pathlib import Path
from itertools import permutations

import numpy as np

from slurm.gcd4da.commons.kmeans import SSKMeansTrainer as KMeans


domain_infos = {
    'hello': [
        ['k400-SVT', 'ucf101', 'ucf'],
        ['k400-SVT', 'hmdb51', 'hmdb']
    ],
    'ek100': [
        ['k400-SVT', 'ek100', 'P02'],
        ['k400-SVT', 'ek100', 'P04'],
        ['k400-SVT', 'ek100', 'P22']
    ],
    'simnreal': [
        ['in1k', 'k400', 'k400'],
        ['in1k', 'babel', 'babel']
    ],
}  # pretrained weight, dataset name, domain name (nickname)

p_repo_root = Path('/data/hyogun/repos/haawron_mmaction2')
p_features_dir = p_repo_root / 'data/features'
p_filelists_dir = p_repo_root / 'data/_filelists'
p_labelmaps_dir = p_repo_root / 'data/labelmaps'

header_msg = f'{"task":9}  {"dom1":>5}  -->  {"dom2":<10} {"#known":6}|#unknown'
print(header_msg)
print()
print('=' * len(header_msg) + '\n')
print()

for task_name, domains in domain_infos.items():
    p_labelmap = p_labelmaps_dir / f'{task_name}.txt'
    with p_labelmap.open() as f:
        labelmap = list(csv.reader(f, delimiter=' '))
        num_known_classes = labelmap.index([])
        num_all_classes = len([line for line in labelmap if line])

    for (
            (weight1, dataset_name1, domain_name1),
            (weight2, dataset_name2, domain_name2)
        ) in permutations(domains, 2):

        if task_name == 'ek100':  # sort by target
            (weight1, dataset_name1, domain_name1), (weight2, dataset_name2, domain_name2) \
                = (weight2, dataset_name2, domain_name2), (weight1, dataset_name1, domain_name1)

        print(f'{task_name+":":9}  {domain_name1:>5}  -->  {domain_name2:<10} {num_known_classes:>6d}|{num_all_classes:2d}\n')

        p_pkl_domain1 = p_features_dir / f'{weight1}/{task_name}/{domain_name1}/train.pkl'
        p_pkl_domain2 = p_features_dir / f'{weight2}/{task_name}/{domain_name2}/train.pkl'

        feat_domain1 = np.array(np.load(p_pkl_domain1, allow_pickle=True))
        feat_domain2 = np.array(np.load(p_pkl_domain2, allow_pickle=True))

        p_ann_domain1 = p_filelists_dir / f'{dataset_name1 + ("/processed" if task_name != "hello" else "")}/filelist_{domain_name1}_train_open_all.txt'
        p_ann_domain2 = p_filelists_dir / f'{dataset_name2 + ("/processed" if task_name != "hello" else "")}/filelist_{domain_name2}_train_open_all.txt'

        with p_ann_domain1.open() as f1, p_ann_domain2.open() as f2:
            ann_domain1 = np.array([int(line[-1]) for line in csv.reader(f1, delimiter=' ')])
            ann_domain2 = np.array([int(line[-1]) for line in csv.reader(f2, delimiter=' ')])

        kmeans = KMeans(ks=num_all_classes, autoinit=False, num_known_classes=num_known_classes, verbose=False)
        kmeans.Xs = {
            'train_source': feat_domain1[ann_domain1<num_known_classes],
            'train_target': feat_domain2,
            'valid': feat_domain2,
            'test': feat_domain2,
        }
        kmeans.anns = {
            'train_source': ann_domain1[ann_domain1<num_known_classes],
            'train_target': ann_domain2,
            'valid': ann_domain2,
            'test': ann_domain2,
        }
        kmeans.train()
        kmeans.display()

        print('\n')
    print('-'*len(header_msg) + '\n\n')
