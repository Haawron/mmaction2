import pickle
from pathlib import Path
from semi_kmeans import KMeans as SemiKMeans
import numpy as np
import pandas as pd
import csv
import argparse
import re
from multiprocessing import Pool


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-r', '--root-dir',
        default=Path(r'work_dirs/sanity/ek100/tsm/GCD4DA/P02_P04/19390__GCD4DA_tsm_ek100_sanity_one_way_phase1/0/20220509-222653'),
        type=Path, help='')
    parser.add_argument(
        '-c', '--create-filelist', action='store_true',
        help='')
    args = parser.parse_args()

    np.random.seed(999)  # numpy & sklearn
    n_tries = 5

    p_root = args.root_dir
    source, target = re.findall(r'/([\w^_]{1,4})[2_-]([\w^_]{1,4})/', str(p_root))[0] or [('P02', 'P04')]
    print(f'Task: {source} -> {target}')
    num_classes = (
        5 if 'P' in source else
        12 if source in ['ucf', 'hmdb'] else
        None
    )
    Xs, anns, p_anns = get_input_and_output(p_root, source, target, num_classes)
    for key in Xs.keys():
        print(f'{key:12s}\t{anns[key].shape} {Xs[key].shape}')

    print(f'\nTraining k-means models n_tries={n_tries} ...', end=' ')
    global_best_model, ks, rows_H, rows_os, rows_os_star, rows_unk = train_wrapper(Xs, anns, num_classes, n_tries)
    print('done\n')

    mi_columns = pd.MultiIndex.from_product([['train_source', 'train_target', 'valid', 'test'], ['H', 'OS', 'OS*', 'UNK']], names=['split', 'mode'])
    records = {k: sum([[row_H[key], row_os[key], row_os_star[key], row_unk[key]] for key in row_os.keys()], []) for k, row_H, row_os, row_os_star, row_unk in zip(ks, rows_H, rows_os, rows_os_star, rows_unk)}
    df = pd.DataFrame.from_dict(records, columns=mi_columns, orient='index')
    with pd.option_context('display.precision', 1):
        print(df)

    if args.create_filelist:
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
    print(f'\n\nCreated a filelist in {p_out}.\n')

def create_centroids(model, p_out):
    with p_out.open('wb') as f_cent:
        pickle.dump(model.centroids, f_cent)
    print(f'Created a centroid file in {p_out}.\n')
    

def get_input_and_output(p_root, source, target, num_classes=5):
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
        p_dir_filelist / domain2dataset[target] / f'filelist_{target}_train_open.txt',
        p_dir_filelist / domain2dataset[target] / f'filelist_{target}_{"valid" if domain2dataset[target] == "epic-kitchens-100" else "val"}_open.txt',
        p_dir_filelist / domain2dataset[target] / f'filelist_{target}_test_open.txt'
    ]
    for split_name, p_ann in zip(
        ['train_source', 'train_target', 'valid', 'test'],
        p_anns
    ):
        with p_ann.open('r') as f:
            reader = csv.reader(f, delimiter=' ')
            ann = [int(line[-1]) for line in reader]
            ann = np.array(ann)
            anns[split_name] = ann

    Xs = {}
    for split_name, p_feature in zip(
        ['train_source', 'train_target', 'valid', 'test'],
        [
            p_root / f'features/{source}_train_open.pkl',
            p_root / f'features/{target}_train_open.pkl',
            p_root / f'features/{target}_val_open.pkl',
            p_root / f'features/{target}_test_open.pkl',
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

    Xs['train_source'] = Xs['train_source'][anns['train_source']<num_classes]
    anns['train_source'] = anns['train_source'][anns['train_source']<num_classes]
    return Xs, anns, p_anns


def train_wrapper(Xs, anns, num_classes=5, n_tries=20):
    ks = list(range(num_classes, num_classes+11)) + [30, 50, 100]
    jobs = []
    for k in ks:
        for _ in range(n_tries):
            jobs.append({
                'k': k,
                'Xs': Xs,
                'anns': anns,
                'num_classes': num_classes,
            })
    # results = {k: {'scores_os': {'test': 0}} for k in ks}
    results = {k: {'scores_H': {'test': 0}} for k in ks}  # init 
    with Pool() as p:
        for model, k, scores_H, scores_os, scores_os_star, scores_unk in p.imap_unordered(worker, jobs):
            # if results[k]['scores_os']['test'] < scores_os['test']:
            if results[k]['scores_H']['test'] <= scores_H['test']:
                results[k] = {
                    'model': model,
                    'scores_H': scores_H,
                    'scores_os': scores_os,
                    'scores_os_star': scores_os_star,
                    'scores_unk': scores_unk,
                }
    p.join()

    rows_H, rows_os, rows_os_star, rows_unk = [], [], [], []
    for k in ks:
        rows_H.append(results[k]['scores_H'])
        rows_os.append(results[k]['scores_os'])
        rows_os_star.append(results[k]['scores_os_star'])
        rows_unk.append(results[k]['scores_unk'])
    # global_best_k = max(ks, key=lambda k: results[k]['scores_os']['test'])
    # global_best_test_score = results[global_best_k]['scores_os']['test']
    global_best_k = max(ks, key=lambda k: results[k]['scores_H']['test'])
    global_best_test_score = results[global_best_k]['scores_H']['test']
    global_best_model = results[global_best_k]['model']
    print(f'\nbest test score {global_best_test_score:.1f} at k={global_best_k}')
    return global_best_model, ks, rows_H, rows_os, rows_os_star, rows_unk


def worker(job:dict):
    return train(**job)


def train(k, Xs, anns, num_classes):
    model = get_semi_kmeans_model(k, Xs, anns, num_classes)
    scores_H = get_model_score(model, Xs, anns, num_classes, 'H', False)
    scores_os = get_model_score(model, Xs, anns, num_classes, 'os', False)  # score dict for each split
    scores_os_star = get_model_score(model, Xs, anns, num_classes, 'os*', False)
    scores_unk = get_model_score(model, Xs, anns, num_classes, 'unk', False)
    return model, k, scores_H, scores_os, scores_os_star, scores_unk


def cosine(X, y, norm_X=None):
    norm_X = norm_X if norm_X is not None else np.einsum('ij,ji->i', X, X.T)  # (X @ X.T).diagnoal()
    sim = X @ y / (norm_X * (y@y))**.5
    sim[(sim>1.)  & (abs(sim-1.)<1e-4)] = 1.
    sim[(sim<-1.) & (abs(sim+1.)<1e-4)] = -1.
    return sim


def get_semi_kmeans_model(k, Xs, anns, num_classes=5):
    known_data = [
        np.where(anns['train_source'] == i)[0].astype(np.int64)
        for i in range(num_classes)
    ]  # data indices for each cluster
    X = np.concatenate([Xs['train_source'], Xs['train_target']], axis=0)
    model = SemiKMeans(k=k, known_data=known_data, alpha=0.3, verbose=False, metric='cosine', tol=1e-7).fit(X)
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


def get_model_score(model, Xs, anns, num_classes=5, mode='os', verbose=True):
    scores = {}
    for split_name in ['train_source', 'train_target', 'valid', 'test']:
        if verbose:
            print(split_name.split('_')[0])

        pred_test = model.predict(Xs[split_name])
        pred_test = np.minimum(num_classes, pred_test, dtype=np.int64)
        conf = confusion_matrix(pred_test, anns[split_name])

        if split_name == 'train_source':
            conf = conf[:num_classes]  # [num_classes, num_classes+1]

        recalls = conf.diagonal() / conf.sum(axis=1)
        score = 100 * (
            recalls[:num_classes].mean() if mode == 'os*'
            else (
                0 if len(recalls) == num_classes or recalls[:num_classes].mean() * recalls[num_classes] == 0
                else 2 / ((1 / recalls[:num_classes].mean()) + (1 / recalls[num_classes]))
            )  if mode == 'H'
            else np.NaN if split_name == 'train_source' and mode in ['os', 'unk']
            else recalls.mean() if mode == 'os'
            else recalls[num_classes].mean() if mode == 'unk'
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
