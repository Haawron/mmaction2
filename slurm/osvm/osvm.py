from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
import pandas as pd

from pathlib import Path
import pickle
import csv
import re
import argparse

from itertools import permutations
from collections import defaultdict
from functools import partial


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
    conf = confusion_matrix(preds, labels, num_classes)
    class_accuracies = conf.diagonal() / conf.sum(axis=1)
    return class_accuracies.mean()


def recall_unknown(preds, labels, num_classes):
    conf = confusion_matrix(preds, labels, num_classes)
    return conf[num_classes-1, num_classes-1] / conf[num_classes-1].sum()


def recall(preds, labels):  # for binary
    return (preds==labels)[labels==1].sum() / (labels==1).sum()


def recall_weighted_average(preds, labels, beta=1.3):
    # unknown detection is beta times as important as known detection
    conf = confusion_matrix(np.where(preds==1, 0, 1), np.where(labels==1, 0, 1))
    return np.average(conf.diagonal() / conf.sum(axis=1), weights=[1/(beta+1), beta/(beta+1)])


def f1(preds, labels, positive_label=1):
    # only for binary classification
    assert np.unique(preds).shape[0] <= 2
    assert np.unique(labels).shape[0] <= 2
    if positive_label not in np.unique(labels):
        positive_label = np.unique(labels)[-1]  # the largest one
    positive_GT = (labels == positive_label).sum()
    positive_pred = (preds == positive_label).sum()
    TP = ((preds == labels) & (labels == positive_label)).sum() 
    if TP == 0:
        return 0
    else:
        recall = TP / positive_GT
        precision = TP / positive_pred
        return 2 / ((1 / precision) + (1 / recall))


def confusion_matrix(preds, labels, num_classes=None):
    h = num_classes or labels.max() + 1
    w = num_classes or preds.max() + 1
    conf = np.bincount(w * labels + preds, minlength=h*w).reshape(h, w)
    return conf


def build_osvm(c, data_train_c, data_valid, label_valid, criterion=f1):
    ss = StandardScaler()
    ss.fit(data_train_c)
    data_train_c = ss.transform(data_train_c)
    data_valid = ss.transform(data_valid)
    assert c in label_valid
    label_valid = np.where(label_valid == c, 1, -1)
    best_osvm = None
    best_valid_score = 0
    # print(f'\n{c}')
    j_nu, j_gamma = 0, 0
    for i_gamma, gamma in enumerate(np.logspace(-3, -1, 20, endpoint=False)):
        for i_nu, nu in enumerate(np.logspace(-4, -.3, 30, endpoint=False)):
            osvm = svm.OneClassSVM(gamma=gamma, kernel='rbf', nu=nu)
            osvm.fit(data_train_c)
            pred = osvm.predict(data_valid)
            score = criterion(pred, label_valid)
            # print(f'gamma: {gamma:.10f}, nu: {nu:.10f}, score: {score:.4f}', end=' ')
            # print(confusion_matrix(np.where(pred==1, 0, 1), np.where(label_valid==1, 0, 1)))
            if score >= best_valid_score:
                # print('selected!')
                best_osvm = osvm
                best_valid_score = score
                j_nu = i_nu
                j_gamma = i_gamma
            else:
                # print()
                pass
    params = best_osvm.get_params()
    print(f'c: {c:2d}, gamma [{j_gamma+1:2d}/{i_gamma+1}]: {params["gamma"]:.7f}, nu [{j_nu+1:2d}/{i_nu+1}]: {params["nu"]:.7f}, valid_score: {best_valid_score:.4f}')
    return best_osvm, ss, best_valid_score


def get_scores_from_osvms(sss, osvms, data, labels, num_classes, unk: bool, metric: str, score_mean=None, score_std=None):
    scores, preds = [], []
    for ss, osvm in zip(sss, osvms):
        data_c = ss.transform(data)
        score = osvm.score_samples(data_c)
        pred = osvm.predict(data_c)
        scores.append(score)
        preds.append(pred)

    scores = np.array(scores)
    preds = np.array(preds)  # predictions for each osvm: C x N, in {-1, 1}^{C x N}

    # standardize
    on_test = score_mean is not None and score_std is not None
    score_mean = score_mean if on_test else scores.mean(axis=1, keepdims=True)
    score_std = score_std if on_test else scores.std(axis=1, keepdims=True)
    scores = (scores - score_mean) / score_std

    final_pred = scores.argmax(axis=0)  # final prediction range(C+1)^N
    pred_unknown = np.logical_and.reduce(preds==-1, axis=0)
    final_pred[pred_unknown] = num_classes

    if unk:
        os = globals()[metric](final_pred, labels, num_classes+1)
        unk = recall_unknown(final_pred, labels, num_classes+1)
        os_star = (num_classes * os - unk) / (num_classes - 1)
        conf = confusion_matrix(final_pred, labels, num_classes+1)

        if on_test:
            return os, os_star, unk, conf
        else:
            return os, os_star, unk, conf, score_mean, score_std

    else:
        os_star = globals()[metric](final_pred, labels)
        conf = confusion_matrix(final_pred, labels)
        if on_test:
            return os_star, conf
        else:
            return os_star, conf, score_mean, score_std
    

parser = argparse.ArgumentParser(description='OSVM')
# parser.add_argument('-smb', '--select-model-by', choices=['acc', 'mca'], default='mca',
#                     help='')
parser.add_argument('-n', '--num-classes', type=int, default=12,
                    help='without unk')
parser.add_argument('-pd', '--p-data-root', type=Path, default=Path('work_dirs/hello/ucf101/vanilla'),
                    help='')
parser.add_argument('-s', '--source', type=str, default='ucf',
                    help='')
parser.add_argument('-t', '--target', type=str, default='hmdb',
                    help='')
args = parser.parse_args()


num_classes = args.num_classes
metric = 'mean_class_accuracy'

with (args.p_data_root / f'{args.source}_train.pkl').open('rb') as f_train, \
    (args.p_data_root / f'{args.target}_val.pkl').open('rb') as f_val, \
    (args.p_data_root / f'{args.target}_test.pkl').open('rb') as f_test:
    data_train = np.array(pickle.load(f_train))
    data_valid = np.array(pickle.load(f_val))
    data_test = np.array(pickle.load(f_test))

domain2name = {'ucf': 'ucf101', 'hmdb': 'hmdb51', 'P02': 'ek100', 'P04': 'ek100', 'P22': 'ek100'}
with Path(rf'data/_filelists/{domain2name[args.source]}/filelist_{args.source}_train_closed.txt').open('r') as f_train, \
    Path(rf'data/_filelists/{domain2name[args.target]}/filelist_{args.target}_val_open.txt').open('r') as f_val, \
    Path(rf'data/_filelists/{domain2name[args.target]}/filelist_{args.target}_test_open.txt').open('r') as f_test:
    label_train = np.array(list(csv.reader(f_train, delimiter=' ')))[:,-1].astype(np.int32)
    label_valid = np.array(list(csv.reader(f_val, delimiter=' ')))[:,-1].astype(np.int32)
    label_test = np.array(list(csv.reader(f_test, delimiter=' ')))[:,-1].astype(np.int32)

print(data_train.shape, label_train.shape)
print(data_valid.shape, label_valid.shape)
print(data_test.shape, label_test.shape)

# train
sss, osvms = [], []
for c in range(num_classes):
    data_train_c = data_train[label_train==c]
    osvm, ss, acc = build_osvm(c, data_train_c, data_valid, label_valid)
    sss.append(ss)
    osvms.append(osvm)

os_star, conf, mean, std = get_scores_from_osvms(sss, osvms, data_train, label_train, num_classes, False, metric)
print(' '*17 + f'OS*(train): {100*os_star:.1f}')
os, os_star, unk, conf = get_scores_from_osvms(sss, osvms, data_valid, label_valid, num_classes, True, metric, mean, std)
print(f'OS(valid): {100*os:.1f}, OS*(valid): {100*os_star:.1f}, UNK(valid): {100*unk:.1f}')
os, os_star, unk, conf = get_scores_from_osvms(sss, osvms, data_test, label_test, num_classes, True, metric, mean, std)
print(f'OS(test):  {100*os:.1f}, OS*(test):  {100*os_star:.1f}, UNK(test):  {100*unk:.1f}')
print(conf)

# =================================================

# num_classes = 5  # without unk
# metric = 'mean_class_accuracy'
# p_osvm_root = Path(rf'work_dirs/train_output/ek100/tsm/osvm')

# domains = ['P02', 'P04', 'P22']
# table = defaultdict(list)
# tasks = []
# for target, source in permutations(domains, 2):  # ordered by target
#     task = f'{source} → {target}'
#     tasks.append(task)
#     print('==============\n', task)
#     print()
#     for model, task_train_dir, task_test_dir in [
#         ['vanilla', f'{source}/{source}', f'{source}/{target}'],
#         ['dann', f'{source}_{target}', f'{source}_{target}'],
#     ]:
#         print(model)

#         p = max(
#             [p for p in (p_osvm_root / model / task_train_dir).glob('*.pkl') if 'train' in p.name],
#             key=lambda p: {k: v for k, v in zip(*list(csv.reader(p.with_name(p.stem.replace('train', 'test') + '.csv').open('r'))))}[metric]
#         )  # pick the best model's train logits with respect to the metric of the "test" set
#         jid = re.findall(r"(\d+)_train", str(p))[0]
#         print(f'JID of the Selected Model: {jid}')
#         with p.open('rb') as f:
#             data_train = np.array(pickle.load(f))
#         p = p_osvm_root / model / task_test_dir / p.name.replace('train', 'valid')
#         with p.open('rb') as f:
#             data_valid = np.array(pickle.load(f))
#         p = p.with_name(p.name.replace('valid', 'test'))
#         with p.open('rb') as f:
#             data_test = np.array(pickle.load(f))

#         p = Path(rf'data/epic-kitchens-100/filelist_{source}_train_closed.txt')
#         with p.open('r') as f:
#             reader = csv.reader(f, delimiter=' ')
#             label_train = np.array(list(reader))[:,-1].astype(np.int32)
#         p = Path(rf'data/epic-kitchens-100/filelist_{target}_valid_open.txt')
#         with p.open('r') as f:
#             reader = csv.reader(f, delimiter=' ')
#             label_valid = np.array(list(reader))[:,-1].astype(np.int32)
#         p = Path(rf'data/epic-kitchens-100/filelist_{target}_test_open.txt')
#         with p.open('r') as f:
#             reader = csv.reader(f, delimiter=' ')
#             label_test = np.array(list(reader))[:,-1].astype(np.int32)

#         # train
#         best_valid_os = 0
#         best_beta = None
#         best_sss, best_osvms = [], []
#         best_valid_score_mean, best_valid_score_std = None, None
#         for beta in np.linspace(.8, 1.3, 3, endpoint=False):
#             print(f'beta: {beta:.2f}')
#             sss, osvms = [], []
#             for c in range(num_classes):
#                 data_train_c = data_train[label_train==c]
#                 osvm, ss, acc = build_osvm(c, data_train_c, data_valid, label_valid, partial(recall_weighted_average, beta=beta))
#                 sss.append(ss)
#                 osvms.append(osvm)
#             valid_os, valid_os_star, valid_unk, valid_conf, valid_score_mean, valid_score_std =\
#                 get_scores_from_osvms(sss, osvms, data_valid, label_valid, num_classes, metric)
#             print(f'OS(valid): {valid_os:.3f}, OS*(valid): {valid_os_star:.3f}, UNK(valid): {valid_unk:.3f}')
#             if valid_os > best_valid_os:
#                 best_valid_os = valid_os
#                 best_beta = beta
#                 best_sss = sss
#                 best_osvms = osvms
#                 best_valid_score_mean = valid_score_mean
#                 best_valid_score_std = valid_score_std

#         # test
#         print(f'best beta: {best_beta:.2f}')
#         os, os_star, unk, conf = get_scores_from_osvms(best_sss, best_osvms, data_test, label_test, num_classes, metric, best_valid_score_mean, best_valid_score_std)
#         print(conf)

#         table[model] += [os, os_star, unk]
#         print(f'OS  {os:.3f}')
#         print(f'OS* {os_star:.3f}')
#         print(f'UNK {unk:.3f}')
#         print()
#     print()

# for model in table:
#     line = np.array(table[model])
#     average_line = line.reshape(len(tasks), -1).mean(axis=0).tolist()
#     table[model] += average_line
# columns = pd.MultiIndex.from_product([tasks + ['Average'], ['OS', 'OS*', 'UNK']], names=['task', ''])
# df = pd.DataFrame(table.values(), index=table.keys(), columns=columns)
# with pd.option_context('display.precision', 1):
#     print(100*df)
