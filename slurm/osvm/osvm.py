from statistics import harmonic_mean
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from scipy.stats import weibull_min, hmean
import numpy as np

from pathlib import Path
import argparse
import pickle
import csv


def main():
    parser = argparse.ArgumentParser(description='OSVM')
    parser.add_argument('-n', '--num-classes', type=int, default=12,
                        help='without unk')
    parser.add_argument('-pf', '--p-feature-root', type=Path, default=Path('work_dirs/hello/ucf101/vanilla'),
                        help='')
    parser.add_argument('-s', '--source', type=str, default='ucf',
                        help='')
    parser.add_argument('-t', '--target', type=str, default='hmdb',
                        help='')
    args = parser.parse_args()
    main_with_args(args.p_feature_root, args.source, args.target, args.num_classes)


def main_with_args(
    p_feature_root:Path, source:str, target:str,
    num_classes:int
):
    # getting data and label
    # train: source closed
    # valid: target closed (cheated)
    # test: target open
    data_train, data_valid, data_test = get_data(p_feature_root, source, target)
    label_train, label_valid, label_test = get_label(source, target)
    # make closed set
    data_train = data_train[label_train<num_classes]
    label_train = label_train[label_train<num_classes]
    data_valid = data_valid[label_valid<num_classes]
    label_valid = label_valid[label_valid<num_classes]

    # pre-train one-class SVMs
    sss, osvms = train_osvms(
        num_classes, 
        data_train, label_train,
        data_valid, label_valid,
    )
    
    # train P_I-SVM (OSVMs)
    params = train_pi_svm(sss, osvms, data_train)  # not need valid: no model selection

    # test P_I-SVM (OSVMs)
    priors = get_priors(label_train)
    (os_star, unk, H, mca, conf), omega = test_pi_svm(num_classes, priors, sss, osvms, data_test, label_test, params)
    
    # eval
    print(f'\nomega: {omega}')
    print(f'H: {100*H:.1f}  |  OS: {100*mca:.1f}  |  OS*: {100*os_star:.1f}  |  UNK: {100*unk:.1f}')
    print(conf)
    print('\n\n')


def get_data(p_feature_root:Path, source:str, target:str):
    print('retrieving features ...', end=' ')
    with (p_feature_root / f'{source}_train_open.pkl').open('rb') as f_train, \
        (p_feature_root / f'{target}_valid_open.pkl').open('rb') as f_val, \
        (p_feature_root / f'{target}_test_open.pkl').open('rb') as f_test:
        data_train = np.array(pickle.load(f_train))
        data_valid = np.array(pickle.load(f_val))
        data_test = np.array(pickle.load(f_test))
    print('done')
    return data_train, data_valid, data_test


def get_label(source:str, target:str):
    print('retrieving labels ...', end=' ')
    domain2name = {'ucf': 'ucf101', 'hmdb': 'hmdb51', 'P02': 'ek100', 'P04': 'ek100', 'P22': 'ek100'}
    with Path(rf'data/_filelists/{domain2name[source]}/filelist_{source}_train_open.txt').open('r') as f_train, \
        Path(rf'data/_filelists/{domain2name[target]}/filelist_{target}_valid_open.txt').open('r') as f_val, \
        Path(rf'data/_filelists/{domain2name[target]}/filelist_{target}_test_open.txt').open('r') as f_test:
        label_train = np.array(list(csv.reader(f_train, delimiter=' ')))[:,-1].astype(np.int32)
        label_valid = np.array(list(csv.reader(f_val, delimiter=' ')))[:,-1].astype(np.int32)
        label_test = np.array(list(csv.reader(f_test, delimiter=' ')))[:,-1].astype(np.int32)
    print('done')
    return label_train, label_valid, label_test


def train_osvms(
        num_classes, 
        data_train, label_train,
        data_valid, label_valid,
):
    print('training osvms ...')
    sss, osvms = [], []
    # gamma = [.5, .5, .2, .2, .2]
    # C = [.1, .1, .1, .1, .1]
    gamma = [.5, .5, .2, .2, .2]
    C = [.5, .1, .1, .1, .1]
    for c in range(num_classes):
        osvm, ss, acc = build_osvm_for_class_c(c, data_train, label_train, data_valid, label_valid, criterion=f1, gamma=gamma[c], C=C[c])
        sss.append(ss)
        osvms.append(osvm)
    print('done')
    return sss, osvms


def train_pi_svm(sss, osvms, data_train):
    print('training p_i-SVM (OSVMs) ...', end=' ')
    # print()
    params = []
    threshold = 0  # over this, detected
    for c, (ss, osvm) in enumerate(zip(sss, osvms)):
        data_train = ss.transform(data_train)
        scores = get_pred_score_from_osvm(osvm, data_train)
        n_positive_support_vectors = (scores[osvm.support_] > threshold).sum()  # (# of positive support vectors)
        with np.printoptions(suppress=True, precision=5, linewidth=1000):
            print(osvm.support_.shape, scores.shape, n_positive_support_vectors, scores[osvm.support_][:10])
        scores = scores[scores > threshold] if (scores > threshold).any() else scores  # positive as detected
        prob_c = 1.5 * n_positive_support_vectors  # 1.5 * (# of positive support vectors)
        # tau: loc
        # lambda: scale
        # kappa: c (in scipy docs)
        kappa_c, tau_c, lambda_c = weibull_min.fit(np.sort(scores)[:int(max(prob_c, 3))])
        params.append([kappa_c, tau_c, lambda_c])
    print('done')
    return params


def test_pi_svm(
    num_classes,
    priors,
    sss, osvms,
    data_test, label_test,
    params,
    delta=.01  # threshold for rejection
):
    print('testing p_i-SVM (OSVMs) ...', end=' ')
    pred_test = num_classes * np.ones(data_test.shape[0], dtype=np.int64)  # all test data are initialized as UNK
    p = []
    for c, ((kappa_c, tau_c, lambda_c), ss, osvm, prior_c) in enumerate(zip(params, sss, osvms, priors)):
        data_test_c = ss.transform(data_test)
        scores_test = get_pred_score_from_osvm(osvm, data_test_c)
        p_i = weibull_min.cdf(scores_test, c=kappa_c, loc=tau_c, scale=lambda_c)
        p_i *= prior_c  # unnormalized posterior probability
        p.append(p_i)
    p = np.array(p).T  # [N, C]
    omega = p.max()
    pred_test = p.argmax(axis=1)

    curr_metric_score = 0
    for delta in np.linspace(-3, 3, 200, endpoint=True):
        pred_test_tmp = pred_test.copy()
        pred_test_tmp[p.max(axis=1) < delta] = num_classes  # reject
        os_star, unk, H, mca, conf = get_metric_scores(num_classes, pred_test_tmp, label_test)
        criterion = H
        if criterion > curr_metric_score:
            curr_metric_score = criterion
            metric_scores = os_star, unk, H, mca, conf

    print('done')
    return metric_scores, omega


def get_metric_scores(num_classes, pred_test, label_test):
    idx_known = label_test < num_classes
    idx_unknown = ~idx_known
    os_star = mean_class_accuracy(pred_test[idx_known], label_test[idx_known], num_classes=num_classes+1)
    unk = recall_unknown(pred_test[idx_unknown], label_test[idx_unknown], num_classes=num_classes+1)
    H = hmean([os_star, unk]) if os_star * unk > 0 else 0
    mca = (num_classes * os_star + unk) / (num_classes + 1)
    conf = confusion_matrix(pred_test, label_test, num_classes=num_classes+1)
    return os_star, unk, H, mca, conf


def build_osvm_for_class_c(c, data_train, label_train, data_valid, label_valid, criterion, gamma=.2, C=.1):
    data_train_c = data_train[label_train==c]
    ss = StandardScaler()
    ss.fit(data_train_c)
    data_train = ss.transform(data_train)
    data_valid = ss.transform(data_valid)
    assert c in label_valid
    bin_label_train = np.where(label_train == c, 1, -1)
    bin_label_valid = np.where(label_valid == c, 1, -1)
    best_osvm = None
    best_valid_score = 0
    j_C, j_gamma = 0, 0
    # for i_gamma, gamma in enumerate(np.logspace(-2, 1., 20//2, endpoint=False)):
    #     for i_C, C in enumerate(np.logspace(-1., 1., 30//3, endpoint=True)):
    for i_gamma, gamma in enumerate([gamma]):
        for i_C, C in enumerate([C]):
            osvm = svm.SVC(gamma=gamma, kernel='rbf', C=C)
            osvm.fit(data_train, bin_label_train)
            pred = osvm.predict(data_valid)
            valid_score = criterion(pred, bin_label_valid)
            if valid_score >= best_valid_score:
                best_osvm = osvm
                best_valid_score = valid_score
                j_C = i_C
                j_gamma = i_gamma
    params = best_osvm.get_params()
    print(f'c: {c:2d}, gamma [{j_gamma+1:2d}/{i_gamma+1}]: {params["gamma"]:.7f}, C [{j_C+1:2d}/{i_C+1}]: {params["C"]:.7f}, valid_score: {best_valid_score:.4f}')
    return best_osvm, ss, best_valid_score


def get_pred_score_from_osvm(osvm, data):
    # return osvm.score_samples(data)
    return osvm.decision_function(data)


def get_priors(label_train):
    _, counts = np.unique(label_train, return_counts=True)
    priors = counts / counts.sum()
    return priors


######################## metrics ########################

def confusion_matrix(preds, labels, num_classes=None):
    h = labels.max() + 1
    w = num_classes or preds.max() + 1
    conf = np.bincount(w * labels + preds, minlength=h*w).reshape(h, w)
    return conf


def top1_acc(preds, labels):
    return (preds == labels).mean()


def mean_class_accuracy(preds, labels, num_classes=None):
    conf = confusion_matrix(preds, labels, num_classes)
    class_recalls = conf.diagonal() / conf.sum(axis=1)
    return class_recalls.mean()


def recall_unknown(preds, labels, num_classes):
    conf = confusion_matrix(preds, labels, num_classes)
    return conf[num_classes-1, num_classes-1] / conf[num_classes-1].sum()


def recall(preds, labels):  # only for binary
    correct = (preds==labels)[labels==1].sum()
    count = (labels==1).sum()
    if correct == 0: return 0
    return correct / count


def precision(preds, labels):  # only for binary
    correct = (preds==labels)[labels==1].sum()
    count = (preds==1).sum()
    if correct == 0: return 0
    return correct / count


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


if __name__ == '__main__':
    main()
