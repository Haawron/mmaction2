import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import csv
import argparse
import os, subprocess
from typing import List, Dict, Tuple, Union
from multiprocessing import Pool
if __name__ == '__main__':
    from slurm.gcd4da.commons.semi_kmeans import KMeans as SemiKMeans
else:
    from .semi_kmeans import KMeans as SemiKMeans
from slurm.utils.commons.patterns import get_full_infodict_from_jid
from mmaction.core import confusion_matrix
from mmaction.datasets.custom_metrics import split_cluster_acc_v2, split_cluster_acc_v2_balanced


SPLIT_NAMES = ['train_source', 'train_target', 'valid', 'test']

METRIC_H, METRIC_OS, METRIC_OS_STAR, METRIC_UNK = 'H', 'OS', 'OS*', 'UNK'
METRIC_ALL, METRIC_OLD, METRIC_NEW = 'ALL', 'Old', 'New'
METRIC_ALL_BALANCED, METRIC_OLD_BALANCED, METRIC_NEW_BALANCED = 'ALL(B)', 'Old(B)', 'New(B)'
VOSUDA_METRIC_NAMES = [METRIC_H, METRIC_OS, METRIC_OS_STAR, METRIC_UNK]
CDAR_METRIC_NAMES = [
    METRIC_ALL, METRIC_OLD, METRIC_NEW,
    METRIC_ALL_BALANCED, METRIC_OLD_BALANCED, METRIC_NEW_BALANCED
]

DOMAINNAME2DATASETNAME = {
    'ucf': 'ucf101',
    'hmdb': 'hmdb51',
    'P02': 'epic-kitchens-100',
    'P04': 'epic-kitchens-100',
    'P22': 'epic-kitchens-100',
}

ScoreMatrix = List[List[float]]  # Splits(4) x Metrics(3|4)
ResultDict = Dict[int, List[Tuple[SemiKMeans, ScoreMatrix]]]
ScoreDict = Dict[int, ScoreMatrix]


def get_args() -> dict:
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
        '-k', '--ks', type=int, nargs='*', default=[22],
        help='')
    args = parser.parse_args()
    args = vars(args)

    info_dict = get_full_infodict_from_jid(args['jid'])

    args['source'] = info_dict['source']
    args['target'] = info_dict['target']

    if args['jid']:
        p_root = Path(info_dict['config']).parent
        p_features = p_root / 'features'
        args['p_features'] = p_features
    elif args.p_features:
        p_root = args.p_features.parent
        p_features = args.p_features
    if not p_features.is_dir() or len(list(p_features.glob('*.pkl'))) < 4:
        extract_features_unless_yet(args['jid'])

    return args


def extract_features_unless_yet(jid:str):
    assert os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None, 'Please extract features first or get a GPU'
    print('Extracting features first ...\n\n\n')
    subprocess.check_call(f'bash slurm/gcd4da/commons/extract_features_ucf-hmdb_jid.sh {jid}', shell=True)
    print('\n\n\ndone.\n\n')


class SSKMeansTrainer:
    def __init__(self,
        # required options: experiment settings
        source:Union[str,None]=None, target:Union[str,None]=None,

        p_features:Path=None,
        num_known_classes:int=12,
        project_name:str='cdar',  # vosuda, cdar

        # hyper-params
        ks:Union[int, List[int]]=22,
        alpha:float=.3,  # known-centroids' update weight
        metric:str='cosine',
        tol:float=1e-7,

        # other options
        seed_of_seeds:int=5236,
        n_tries:int=20,
        verbose=False,
        autoinit=True,
        **kwargs
    ):
        self.p_features = p_features
        self.source, self.target = source, target
        self.num_known_classes = num_known_classes
        self.project_name = project_name

        self.n_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))

        self.ks = ks if isinstance(ks, (list, tuple)) else [ks]
        self.alpha = alpha
        self.metric = metric
        self.tol = tol

        self.seed_of_seeds = seed_of_seeds
        self.n_tries = n_tries
        self.verbose = verbose

        self.metric_names = {
            'vosuda': VOSUDA_METRIC_NAMES,
            'cdar': CDAR_METRIC_NAMES,
        }[self.project_name]

        if autoinit:
            self._get_inputs_and_anns()

    def train(self):
        jobs = self._generate_jobs()
        results:ResultDict = self._train_all_kmeans_models(jobs)
        self._calculate_score_statistics_from_results_and_draw_best_model(results)

    def display(self):
        mi_columns = pd.MultiIndex.from_product([SPLIT_NAMES, self.metric_names], names=['split', 'k'])
        df_best = pd.DataFrame.from_dict({k: scores.reshape(-1) for k, scores in self.scores_best.items()}, columns=mi_columns, orient='index')
        df_mean = pd.DataFrame.from_dict({k: scores.reshape(-1) for k, scores in self.scores_mean.items()}, columns=mi_columns, orient='index')
        df_std = pd.DataFrame.from_dict({k: scores.reshape(-1) for k, scores in self.scores_std.items()}, columns=mi_columns, orient='index')
        with pd.option_context('display.float_format', '{:5.1f}'.format):
            print('best')
            print(100*df_best)
            print()
            print('mean')
            print(100*df_mean)
            print('std')
            print(100*df_std)
        return df_best, df_mean, df_std

    def get_scores(self):
        return self.scores_best[self.k_best][-1,:3]  # test, best k, ALL/Old/New

    def save(self,df_best, df_mean, df_std):
        p_savedir = self.p_features.parent / 'sskmeans'
        p_savedir.mkdir(exist_ok=True)
        df_best.to_csv(p_savedir / 'best.csv')
        df_mean.to_csv(p_savedir / 'mean.csv')
        df_std.to_csv(p_savedir / 'std.csv')
        print(f'\n\nSaved the results in {str(p_savedir)}')

    def write_results(self):
        # with self.p_features.open('w') as f:
        #     writer = csv.writer(f)
        pass


    def _get_inputs_and_anns(self) -> None:  # self.Xs, self.anns
        anns = {}
        p_dir_filelist = Path(r'data/_filelists')
        p_anns = [
            p_dir_filelist / DOMAINNAME2DATASETNAME[self.source] / f'filelist_{self.source}_train_open.txt',  # to extract only closed
            p_dir_filelist / DOMAINNAME2DATASETNAME[self.target] / f'filelist_{self.target}_train_open_all.txt',
            p_dir_filelist / DOMAINNAME2DATASETNAME[self.target] / f'filelist_{self.target}_{"valid" if DOMAINNAME2DATASETNAME[self.target] == "epic-kitchens-100" else "val"}_open_all.txt',
            p_dir_filelist / DOMAINNAME2DATASETNAME[self.target] / f'filelist_{self.target}_test_open_all.txt'
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
                self.p_features / f'{self.source}_train_open.pkl',
                self.p_features / f'{self.target}_train_open.pkl',
                self.p_features / f'{self.target}_val_open.pkl',
                self.p_features / f'{self.target}_test_open.pkl',
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

        if self.project_name == 'vosuda':
            # to make closed
            Xs['train_source'] = Xs['train_source'][anns['train_source']<self.num_known_classes]
            anns['train_source'] = anns['train_source'][anns['train_source']<self.num_known_classes]

        self.Xs:Dict[str,np.ndarray] = Xs
        self.anns:Dict[str,np.ndarray] = anns

    def _generate_jobs(self) -> list:
        jobs = []
        for k in self.ks:
            for n in range(self.n_tries):
                jobs.append({
                    'k': k,
                    'seed': self.seed_of_seeds * n,
                })
        return jobs

    def _train_all_kmeans_models(self, jobs) -> ResultDict:
        results = defaultdict(list)  # |K| : N_tries x (1 + |SPLITS|x|METRICS|)
        with Pool(self.n_cpus) as p:
            for model, scores in p.imap_unordered(self._worker, jobs):
                results[model.k].append((model, scores))
        p.join()
        return results

    def _worker(self, job):
        model = self._train_single_kmeans_model(**job)
        scores_model:ScoreMatrix = self._get_scores_model(model)
        return model, scores_model

    def _train_single_kmeans_model(self, k, seed) -> SemiKMeans:
        known_data = [
            np.where(self.anns['train_source'] == i)[0].astype(np.int64)
            for i in range(self.num_known_classes)
        ]  # data indices for each cluster
        X = np.concatenate([self.Xs['train_source'], self.Xs['train_target']], axis=0)
        model = SemiKMeans(
            k=k, known_data=known_data,
            alpha=self.alpha, metric=self.metric, tol=self.tol,
            seed=seed,
            verbose=False).fit(X)
        return model

    def _get_scores_model(self, model) -> ScoreMatrix:
        scores_model:ScoreMatrix = []
        pred_tests:dict = self.predict(model)
        for split_name in SPLIT_NAMES:
            if self.anns[split_name] is None:
                continue

            conf = confusion_matrix(pred_tests[split_name], self.anns[split_name])
            if self.project_name == 'vosuda':
                scores_for_this_split:List[float] = [self._get_score_from_confmat(conf, self.num_known_classes, metric_name) for metric_name in self.metric_names]
            elif self.project_name == 'cdar':
                if split_name == 'train_source':
                    scores_for_this_split:List[float] = [0, 0, 0, 0, 0, 0]
                else:
                    scores_for_this_split:List[float] = self._get_gcd_scores(pred_tests[split_name], self.anns[split_name], self.num_known_classes)
            scores_model.append(scores_for_this_split)

            if self.verbose:
                print(split_name.split('_')[0])
                with np.printoptions(threshold=np.inf, linewidth=1000):
                    print(conf)
                print(' '.join([f'{score:.1f}' for score in scores_for_this_split]))
                print()
        return scores_model

    def predict(self, model) -> Dict[str,Union[None,np.ndarray]]:
        pred_tests = {}
        for split_name, X in self.Xs.items():
            if X is not None:
                pred_test = model.predict(X)
                if self.project_name == 'vosuda':
                    pred_test = np.minimum(self.num_known_classes, pred_test, dtype=np.int64)
                pred_tests[split_name] = pred_test
            else:
                pred_tests[split_name] = None
        return pred_tests

    def _get_score_from_confmat(self, conf, num_known_classes, metric_name:str=METRIC_H):
        if conf.shape[0] > num_known_classes + 1:
            conf[num_known_classes] = conf[num_known_classes:].sum(axis=0)
            conf = conf[:num_known_classes+1]
        if conf.shape[1] > num_known_classes + 1:
            conf[:,num_known_classes] = conf[:,num_known_classes:].sum(axis=1)
            conf = conf[:,:num_known_classes+1]

        correct = conf.diagonal()  # [K+1]
        label_dist = conf.sum(axis=1)  # [K+1]
        label_dist[label_dist==0] += 1  # avoiding true_divide warning(occurs when evaluated with closed set)
        recalls = correct / label_dist

        def H(recalls):
            if (
                recalls.shape[0] == num_known_classes
                or recalls[:num_known_classes].mean() == 0
                or recalls[num_known_classes] == 0
            ):
                return 0
            else:
                return 2 / ((1 / recalls[:num_known_classes].mean()) + (1 / recalls[num_known_classes]))

        def OS(recalls):
            if recalls.shape[0] == num_known_classes:
                return np.NaN
            return recalls.mean()

        def OS_STAR(recalls):
            return recalls[:num_known_classes].mean()

        def UNK(recalls):
            if recalls.shape[0] == num_known_classes:
                return np.NaN
            return recalls[-1]

        metrics = [H, OS, OS_STAR, UNK]
        score = 100 * metrics[self.metric_names.index(metric_name)](recalls)
        return score

    def _get_gcd_scores(self, pred, gt, num_known_classes) -> List[float]:
        old_mask = (gt < num_known_classes)
        scores = []
        total_acc, old_acc, new_acc = split_cluster_acc_v2(gt, pred, old_mask)
        scores.extend([total_acc, old_acc, new_acc])
        total_acc, old_acc, new_acc = split_cluster_acc_v2_balanced(gt, pred, old_mask)
        scores.extend([total_acc, old_acc, new_acc])
        return scores

    def _calculate_score_statistics_from_results_and_draw_best_model(self, results:ResultDict) -> None:
        self.scores_best:ScoreDict = {}
        self.scores_mean:ScoreDict = {}
        self.scores_std:ScoreDict = {}

        self.model_best = None
        self.k_best = -1
        score_best = 0
        for k, result in results.items():
            models, all_scores = zip(*result)  # [N_tries], [N_tries, |Splits|, |Metrics|]
            arg_best:int = np.argmax(all_scores, axis=0)[-1,0]  # test, H
            self.scores_best[k] = np.array(all_scores[arg_best])  # [|Splits|, |Metrics|]
            self.scores_mean[k] = np.mean(all_scores, axis=0)  # [|Splits|, |Metrics|]
            self.scores_std[k] = np.std(all_scores, axis=0)  # [|Splits|, |Metrics|]

            score_current_test_best = self.scores_best[k][-1,0]  # test, H
            if score_current_test_best > score_best:
                score_best = score_current_test_best
                self.model_best = models[arg_best]
                self.k_best = k


if __name__ == '__main__':
    np.random.seed(999)  # numpy & sklearn

    args:dict = get_args()

    trainer = SSKMeansTrainer(**args)
    trainer.train()
    trainer.display()
    trainer.write_results()
