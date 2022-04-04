from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
import pandas as pd

from pathlib import Path
import pickle
import csv
import re
from itertools import permutations
from collections import defaultdict


# given jid

# if jid is vanilla's
    # for domain_tested in P02 P04 P22
        # assert task: source-only
        # extract logits from train_closed dataset (with test.py)
        # save logits to work_dirs/train_output/ek100/tsm/osvm/vanilla/P02/{domain_tested}/jid.pkl
# else
    # extract logits from train_closed dataset (with test.py)
    # save logits to work_dirs/train_output/ek100/tsm/osvm/dann/P02_P04/jid.pkl
# ==========================================
# load logits
# train osvm
# test on valid_open
# select the best model
# test on test_open

def top1_acc(preds, labels):
    return (preds == labels).mean()


def mean_class_accuracy(preds, labels, num_classes=None):
    num_classes = num_classes or labels.max()
    conf = confusion_matrix(preds, labels, num_classes)
    class_accuracies = conf.diagonal() / conf.sum(axis=1)
    return class_accuracies.mean()


def confusion_matrix(preds, labels, num_classes=None):
    num_classes = num_classes or labels.max()
    conf = np.bincount(num_classes * labels + pred, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf


num_classes = 5
metric = 'mean_class_accuracy'
p_osvm_root = Path(rf'work_dirs/train_output/ek100/tsm/osvm')

domains = ['P02', 'P04', 'P22']
table = defaultdict(list)
tasks = []
for target, source in permutations(domains, 2):  # ordered by target
    task = f'{source} → {target}'
    tasks.append(task)
    print('==============\n', task)
    print()
    for model, task_train_dir, task_test_dir in [
        ['vanilla', f'{source}/{source}', f'{source}/{target}'],
        ['dann', f'{source}_{target}', f'{source}_{target}'],
    ]:
        print(model)
        p = max(
            [p for p in (p_osvm_root / model / task_train_dir).glob('*.pkl') if 'train' in p.name],
            key=lambda p: {k: v for k, v in zip(*list(csv.reader(p.with_suffix('.csv').open('r'))))}[metric]
        )
        jid = re.findall(r"(\d+)_train", str(p))[0]
        print(f'JID of the Selected Model: {jid}')
        with p.open('rb') as f:
            data = np.array(pickle.load(f))

        p = p_osvm_root / model / task_test_dir / p.name.replace('train', 'test')
        with p.open('rb') as f:
            data_test = np.array(pickle.load(f))

        p = Path(rf'data/epic-kitchens-100/filelist_{source}_train_closed.txt')
        with p.open('r') as f:
            reader = csv.reader(f, delimiter=' ')
            train_labels = np.array(list(reader))[:,-1].astype(np.int32)

        p = Path(rf'data/epic-kitchens-100/filelist_{target}_test_open.txt')
        with p.open('r') as f:
            reader = csv.reader(f, delimiter=' ')
            test_labels = np.array(list(reader))[:,-1].astype(np.int32)

        bests = {'OS': 0, 'OS*': 0}
        for nu in np.linspace(.2, .9, 20):
            osvms = []
            scores, preds = [], []
            for c in range(num_classes):
                data_c = data[train_labels==c]
                ss = StandardScaler()
                ss.fit(data_c)
                data_c = ss.transform(data_c)
                osvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=nu)
                osvm.fit(data_c)
                osvms.append(osvm)

                data_test_c = ss.transform(data_test)
                score = osvm.score_samples(data_test_c)
                pred = osvm.predict(data_test_c)
                scores.append(score)
                preds.append(pred)
                
            scores = np.array(scores)
            preds = np.array(preds)  # predictions for each osvm {-1, 1}^N

            result = scores.argmax(axis=0)  # final prediction range(C+1)^N
            assert result.shape == test_labels.shape
            os_star = (result == test_labels)[test_labels != num_classes].mean()
            bests['OS*'] = max(os_star, bests['OS*'])

            unknown = np.logical_and.reduce(preds==-1, axis=0)
            result[unknown] = num_classes
            os = (result == test_labels).mean()
            bests['OS'] = max(os, bests['OS'])
        table[model] += [bests['OS'], bests['OS*']]
        print(f"OS  {bests['OS']:.3f}")
        print(f"OS* {bests['OS*']:.3f}")
        print()
    print()

columns = pd.MultiIndex.from_product([tasks, ['OS', 'OS*']], names=['task', ''])
df = pd.DataFrame(table.values(), index=table.keys(), columns=columns)
with pd.option_context('display.precision', 1):
    print(100*df)
