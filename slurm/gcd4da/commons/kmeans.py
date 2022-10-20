import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import csv
import argparse
import re
import os
from multiprocessing import Pool
if __name__ == '__main__':
    from slurm.gcd4da.commons.semi_kmeans import KMeans as SemiKMeans
else:
    from .semi_kmeans import KMeans as SemiKMeans
from slurm.utils.commons.patterns import get_full_infodict_from_jid


SPLIT_NAMES = ['train_source', 'train_target', 'valid', 'test']

METRIC_H, METRIC_OS, METRIC_OS_STAR, METRIC_UNK = 'H', 'OS', 'OS*', 'UNK'
METRIC_ALL, METRIC_OLD, METRIC_NEW = 'ALL', 'Old', 'New'
VOSUDA_METRICS = [METRIC_H, METRIC_OS, METRIC_OS_STAR, METRIC_UNK]
CDAR_METRICS = [METRIC_ALL, METRIC_OLD, METRIC_NEW]


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-j', '--jid', default=None, type=str,
        help='')
    parser.add_argument(
        '-f', '--p-features', default=None, type=Path,
        help='')

    parser.add_argument(
        '-c', '--create-filelist', action='store_true',
        help='')
    parser.add_argument(
        '-n', '--n-tries', type=int, default=5,
        help='')
    parser.add_argument(
        '-k', '--fixed-k', type=int, default=None,
        help='')
    args = parser.parse_args()

    np.random.seed(999)  # numpy & sklearn

    assert (args.jid is not None) ^ (args.p_features is not None)  # exactly one

    if args.jid:
        p_root = Path(get_full_infodict_from_jid(args.jid)['config']).parent
    elif args.p_features:
        p_root = args.p_features.parent
    source, target = re.findall(r'/([\w^_]{1,4})[2_-]([\w^_]{1,4})/', str(p_root))[0] or [('P02', 'P04')]
    print(f'Task: {source} -> {target}')
    num_classes = (
        5 if 'P' in source else
        12 if source in ['ucf', 'hmdb'] else
        None
    )

    p_features = args.p_features if args.p_features else p_root / 'features'
    Xs, anns, p_anns = get_input_and_output(p_features, source, target, num_classes)
    for key in Xs.keys():
        print(f'{key:12s}\t{anns[key].shape} {Xs[key].shape}')

    print(f'\nTraining k-means models n_tries={args.n_tries} ...')
    global_best_model, ks, score_rows = train_wrapper(Xs, anns, num_known_classes=num_classes, fixed_k=args.fixed_k, n_tries=args.n_tries)
    print('done\n')

    mi_columns = pd.MultiIndex.from_product([SPLIT_NAMES, VOSUDA_METRICS], names=['split', 'mode'])
    df = pd.DataFrame.from_dict({k: row for k, row in zip(ks, score_rows)}, columns=mi_columns, orient='index')
    with pd.option_context('display.precision', 1):
        print(df)

    if args.create_filelist:
        print('\n\n')
        p_out = p_root / f'filelist_pseudo_{target}_k{global_best_model.k:03d}.txt'
        create_filelist(
            global_best_model,
            Xs['train_target'],
            p_anns[1],
            p_out
        )

        p_out = p_out.with_name(f'filelist_test_{target}_k{global_best_model.k:03d}.txt')
        create_filelist(
            global_best_model,
            Xs['test'],
            p_anns[3],
            p_out
        )

        p_out = p_out.with_name('centroids.pkl')
        create_centroids(global_best_model, p_out)


def create_filelist(model, X_train_target, p_ann_train_target, p_out):
    pseudo_labels = model.predict(X_train_target)
    with p_ann_train_target.open('r') as f_ann, p_out.open('w') as f_out:
        reader = csv.reader(f_ann, delimiter=' ')
        writer = csv.writer(f_out, delimiter=' ')
        for line, pseudo_label in zip(reader, pseudo_labels):
            writer.writerow(line[:-1] + [pseudo_label])
    print(f'Created a filelist in "{p_out}".\n')

def create_centroids(model, p_out):
    with p_out.open('wb') as f_cent:
        pickle.dump(model.centroids, f_cent)
    print(f'Created a centroid file in "{p_out}".\n')


def get_input_and_output(p_feature, source, target, num_classes=5):
    anns = {}
    domain2dataset = {
        'ucf': 'ucf101',
        'hmdb': 'hmdb51',
        'P02': 'epic-kitchens-100',
        'P04': 'epic-kitchens-100',
        'P22': 'epic-kitchens-100',
    }
    p_dir_filelist = Path(r'data/_filelists')
    p_anns = [
        p_dir_filelist / domain2dataset[source] / f'filelist_{source}_train_open.txt',  # to extract only closed
        p_dir_filelist / domain2dataset[target] / f'filelist_{target}_train_open_all.txt',
        p_dir_filelist / domain2dataset[target] / f'filelist_{target}_{"valid" if domain2dataset[target] == "epic-kitchens-100" else "val"}_open_all.txt',
        p_dir_filelist / domain2dataset[target] / f'filelist_{target}_test_open_all.txt'
    ]
    for split_name, p_ann in zip(SPLIT_NAMES, p_anns):
        with p_ann.open('r') as f:
            reader = csv.reader(f, delimiter=' ')
            ann = [int(line[-1]) for line in reader]
            ann = np.array(ann)
            anns[split_name] = ann

    Xs = {}
    for split_name, p_feature in zip(
        SPLIT_NAMES, [
            p_feature / f'{source}_train_open.pkl',
            p_feature / f'{target}_train_open.pkl',
            p_feature / f'{target}_val_open.pkl',
            p_feature / f'{target}_test_open.pkl',
        ]
    ):
        try:
            f = p_feature.open('rb')
        except FileNotFoundError:
            f = p_feature.with_name(p_feature.name.replace('val', 'valid')).open('rb')
        X = pickle.load(f)
        X = np.array(X)
        Xs[split_name] = X
        f.close()

    # to make closed
    Xs['train_source'] = Xs['train_source'][anns['train_source']<num_classes]
    anns['train_source'] = anns['train_source'][anns['train_source']<num_classes]
    return Xs, anns, p_anns


def train_wrapper(Xs, anns, num_known_classes=5, fixed_k:int=None, n_tries=20, verbose=True):
    print([np.linalg.norm(X) for X in Xs.values()])
    if fixed_k is None:
        ks = set(range(num_known_classes, num_known_classes+20)) | {30, 50, 100}
        ks = sorted(ks)
    else:
        assert fixed_k >= num_known_classes
        ks = [fixed_k]
    jobs = []
    for k in ks:
        for n in range(n_tries):
            jobs.append({
                'k': k,
                'Xs': Xs,
                'anns': anns,
                'num_classes': num_known_classes,
                'seed': 5236 * n,
            })
    # results = {
    #     12: {  # k
    #         'scores_H': [
    #             [88.7, 88.3, 75.4, 77.8],  # train_source, train_target, valid, test
    #             [88.7, 88.3, 75.4, 77.8],
    #             ...
    #         ],
    #         'scores_os': [
    #             ...
    #         ],
    #         'scores_os_star': [
    #             ...
    #         ],
    #         'scores_unk': [
    #             [88.7, 88.3, 75.4, 77.8],  # train_source, train_target, valid, test
    #             [88.7, 88.3, 75.4, 77.8],
    #             ...
    #         ],
    #     },
    #     13: {
    #       ...
    #     },
    #     ...
    # }
    # dict of dicts of lists of dicts
    results = defaultdict(lambda: defaultdict(list))
    global_best_test_score = 0
    global_best_test_scores = []
    global_best_k = None
    with Pool(int(os.environ.get('SLURM_CPUS_ON_NODE', 8))) as p:
        for model, k, scores in p.imap_unordered(worker, jobs):
            if global_best_test_score < scores[0]['test']:  # scores[0]: head metric
                global_best_test_score = scores[0]['test']
                global_best_test_scores = [f'{score["test"]:.1f}' for score in scores]
                global_best_model = model
                global_best_k = k
            for metric, score in zip(VOSUDA_METRICS, scores):
                results[k][metric].append(list(score.values()))
    p.join()

    # rows = [
    #     [33.7, 33.7, 33.7, 33.7, ..., 33.7, 33.7, 33.7, 33.7],  # [train_source, train_target, valid, test] x [H, OS, OS*, UNK]
    #     ...  # for all k's
    # ]
    rows = []
    for k in ks:
        means = np.concatenate([
            np.mean(results[k][metric], axis=0, keepdims=True)
            for metric in VOSUDA_METRICS
        ], axis=0).T.reshape(-1).tolist()  # [16]
        # stddevs = np.concatenate([
        #     np.std(results[k]['scores_H'], axis=0, keepdims=True),
        #     np.std(results[k]['scores_os'], axis=0, keepdims=True),
        #     np.std(results[k]['scores_os_star'], axis=0, keepdims=True),
        #     np.std(results[k]['scores_unk'], axis=0, keepdims=True),
        # ], axis=0).T.reshape(-1)  # 5 ~ 7 정도 남
        rows.append(means)
    # TODO: global best model은 뭘 보고 뽑음? best k -> best model?
    if verbose:
        print(f'\nbest test score {global_best_test_score:.1f} at k={global_best_k}')
        print('\t', ' '.join(global_best_test_scores))
        global_best_mean_test_score, global_best_mean_test_k = max(zip([means[-4] for means in rows], ks), key=lambda zipped_row: zipped_row[0])
        print(f'best (mean) test score {global_best_mean_test_score:.1f} at k={global_best_mean_test_k}')
    return global_best_model, ks, rows


def worker(job:dict):
    return train(**job)


def train(k, Xs, anns, num_classes, seed):
    model = get_semi_kmeans_model(k, Xs, anns, num_classes, seed)
    scores = [get_model_score(model, Xs, anns, num_classes, metric, False) for metric in VOSUDA_METRICS]
    return model, k, scores


def cosine(X, y, norm_X=None):
    norm_X = norm_X if norm_X is not None else np.einsum('ij,ji->i', X, X.T)  # (X @ X.T).diagnoal()
    sim = X @ y / (norm_X * (y@y))**.5
    sim[(sim>1.)  & (abs(sim-1.)<1e-4)] = 1.
    sim[(sim<-1.) & (abs(sim+1.)<1e-4)] = -1.
    return sim


def get_semi_kmeans_model(k, Xs, anns, num_classes=5, seed=999):
    known_data = [
        np.where(anns['train_source'] == i)[0].astype(np.int64)
        for i in range(num_classes)
    ]  # data indices for each cluster
    X = np.concatenate([Xs['train_source'], Xs['train_target']], axis=0)
    model = SemiKMeans(k=k, known_data=known_data, alpha=0.3, verbose=False, metric='cosine', tol=1e-7, seed=seed).fit(X)
    return model


def get_kmeans_model(k, Xs, anns, num_classes=5):
    from sklearn.cluster import KMeans
    # 1. get initial centroids
    # 1-1. semi. k-means: centroids are mean vector for each class for labeled data
    centroids = []
    centroid_args = []
    for c in range(num_classes):
        ann_train_source_closed = anns['train_source'][anns['train_source'] != num_classes]
        centroids.append(Xs['train_source'][ann_train_source_closed==c].mean(axis=0))
    # 1-2. k-means++: centroids are initialized by random choices w.p. their distances to the last picked centroid
    new_centroid = centroids[-1]
    X = np.concatenate((Xs['train_source'], Xs['train_target']), axis=0)
    norm_X = np.einsum('ij,ji->i', X, X.T)  # (X @ X.T).diagnoal()
    for _ in range(k-num_classes):
        y = new_centroid
        sim = cosine(X, y, norm_X)
        assertion = ((-1 <= sim) & (sim <= 1))
        assert assertion.all(), sim[~assertion]
        d = np.arccos(sim)
        d **= .3  # Heuristics: d = theta ** .3
        d[centroid_args] = 0
        centroid_arg = np.random.choice(X.shape[0], p=d/d.sum())
        centroid_args.append(centroid_arg)
        new_centroid = X[centroid_arg]
        centroids.append(new_centroid)
    centroids = np.array(centroids)
    # 2. train
    model = KMeans(n_clusters=k, init=centroids, n_init=1, tol=1e-7).fit(X)
    return model


def _get_model_score_(confmat, mode=METRIC_H):
    pass

def get_model_score(model, Xs, anns, num_classes=5, mode=METRIC_H, verbose=True):
    assert mode in VOSUDA_METRICS + CDAR_METRICS
    scores = {}
    for split_name in SPLIT_NAMES:
        if Xs[split_name] is None or anns[split_name] is None:
            continue
        if verbose:
            print(split_name.split('_')[0])

        pred_test = model.predict(Xs[split_name])
        pred_test = np.minimum(num_classes, pred_test, dtype=np.int64)
        conf = confusion_matrix(pred_test, anns[split_name])

        if mode in VOSUDA_METRICS:
            if conf.shape[0] > num_classes:
                conf[num_classes] = conf[num_classes:].sum(axis=0)
                conf = conf[:num_classes+1]
            if conf.shape[1] > num_classes:
                conf[:,num_classes] = conf[:,num_classes:].sum(axis=1)
                conf = conf[:,:num_classes+1]

        if split_name == 'train_source':
            conf = conf[:num_classes]  # [num_classes, num_classes+1]

        recalls = conf.diagonal() / conf.sum(axis=1)
        score = 100 * (
            recalls[:num_classes].mean() if mode == METRIC_OS_STAR
            else (
                0 if len(recalls) == num_classes or recalls[:num_classes].mean() * recalls[num_classes] == 0
                else 2 / ((1 / recalls[:num_classes].mean()) + (1 / recalls[num_classes]))
            )  if mode == METRIC_H
            else np.NaN if split_name == 'train_source' and mode in [METRIC_OS, METRIC_UNK]
            else recalls.mean() if mode == METRIC_OS
            else recalls[num_classes].mean() if mode == METRIC_UNK
            else np.NaN
        )
        scores[split_name] = score
        if verbose:
            print(conf)
            print(f'{score:.1f}')
            print()
    return scores


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
        if y_pred.dtype == np.int32:
            y_pred = y_pred.astype(np.int64)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
        if y_real.dtype == np.int32:
            y_real = y_real.astype(np.int64)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


if __name__ == '__main__':
    main()
