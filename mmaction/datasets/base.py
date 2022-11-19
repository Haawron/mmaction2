# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from pathlib import Path
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import os
import re
import mmcv
import numpy as np
import torch
import time
import pickle
from mmcv.utils import print_log
from torch.utils.data import Dataset

from slurm.gcd4da.commons.kmeans_pre import train_wrapper as train_sskmeans

from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy,
                    confusion_matrix)
from .pipelines import Compose

# from slurm.gcd4da.commons.kmeans import train


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            # metric_options['top_k_accuracy'] = dict(
            #     metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'H_mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision', 'recall_unknown', 'confusion_matrix', 'kmeans', 'sskmeans', 'logits'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        if metric_options.get('use_predefined_labels', False):
            gt_labels = self.predefined_labels
        else:
            gt_labels = [ann['label'] for ann in self.video_infos]

        results = np.array(results)
        # 이거 왜 있지?
        # if self.num_classes:
        #     if results.shape[1] > self.num_classes:
        #         results = np.hstack([results[:,:self.num_classes-1], results[:,self.num_classes-1:].max(axis=1, keepdims=True)])

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'H_mean_class_accuracy':  # only valid for UNK
                pred = np.argmax(results, axis=1)
                cf_mat = confusion_matrix(pred, gt_labels)
                cls_cnt = cf_mat.sum(axis=1)
                cls_hit = np.diag(cf_mat)
                cls_acc = np.array([hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])
                os_star, unk = cls_acc[:-1].mean(), cls_acc[-1]
                H_mean_acc = 2 * os_star * unk / (os_star + unk)
                eval_results['H_mean_class_accuracy'] = H_mean_acc
                eval_results['os*'] = os_star
                eval_results['recall_unknown'] = unk
                log_msg = f'\nH_mean_acc\t{H_mean_acc:.4f} (OS* {os_star:.4f}, UNK: {unk:.4f})'
                print_log(log_msg, logger=logger)
                continue

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results,
                                                      gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'recall_unknown':
                pred = np.argmax(results, axis=1)
                conf = confusion_matrix(pred, gt_labels)
                recall = conf[-1,-1] / conf[-1,:].sum()
                eval_results['recall_unknown'] = recall
                log_msg = f'\nrecall_unknown\t{recall:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'confusion_matrix':
                pred = np.argmax(results, axis=1)
                conf = confusion_matrix(pred, gt_labels)
                h, w = conf.shape
                with np.printoptions(threshold=np.inf, linewidth=100000):
                    s = str(conf)
                if 'SRUN_DEBUG' in os.environ:  # if in srun(debuging) session
                    for start, end in reversed([(m.start(0), m.end(0)) for i, m in enumerate(re.finditer(r'\d+', str(s))) if i%h == i//w]):
                        s = s[:start] + '\033[1m' + s[start:end] + '\033[0m' + s[end:]  # diag vals to be bold
                log_msg = '\n' + s
                print_log(log_msg, logger=logger)
                continue

            if metric == 'kmeans':
                pass

            if metric == 'sskmeans':
                # to resolve circular import
                from mmaction.datasets.dataset_wrappers import ConcatDataset
                assert type(self) == ConcatDataset
                gt_labels = np.array(gt_labels)
                Xs = {
                    'train_source': results[:self.cumsum[0]],
                    'train_target': results[self.cumsum[0]:self.cumsum[1]],
                    'valid': results[self.cumsum[1]:self.cumsum[2]],
                    'test': results[self.cumsum[2]:],
                }
                anns = {
                    'train_source': gt_labels[:self.cumsum[0]],
                    'train_target': None,
                    'valid': gt_labels[self.cumsum[1]:self.cumsum[2]],  # cheat
                    'test': gt_labels[self.cumsum[2]:],
                }
                metric_option = metric_options.get('sskmeans', {})
                n_tries = metric_option.get('n_tries', 1)  # default = {'sskmeans': {'n_tries': 1}}
                fixed_k = metric_option.get('fixed_k', None)
                _, ks, rows = train_sskmeans(
                    Xs, anns, num_known_classes=self.num_classes,
                    fixed_k=fixed_k, n_tries=n_tries, verbose=False
                )
                if fixed_k:
                    # ks = [fixed_k]
                    global_best_mean_test_score, os_star, unk = rows[0][-4], rows[0][-2], rows[0][-1]
                    log_msg = f'\nSS k-means H\t{global_best_mean_test_score:.4f}\n{"OS*":>14s}\t{os_star:.4f}\n{"UNK":>14s}\t{unk:.4f} (fixed_k: {fixed_k})'
                else:
                    global_best_mean_test_score, global_best_mean_test_k = max(zip([means[-4] for means in rows], ks), key=lambda zipped_row: zipped_row[0])
                    log_msg = f'\nSS k-means (H)\t{global_best_mean_test_score:.4f} (best_k: {global_best_mean_test_k})'
                eval_results[metric] = global_best_mean_test_score
                eval_results['os*'] = os_star
                eval_results['recall_unknown'] = unk
                print_log(log_msg, logger=logger)
                continue

            if metric == 'logits':
                p_out_dir = metric_options.get('logits', {}).get('p_out_dir', None)
                assert p_out_dir is not None, "Specify the out dir in metric_options['logits']['p_out_dir']"
                p_out = Path(p_out_dir) / 'logits' / f'{int(time.time())}.pkl'
                p_out.parent.mkdir(exist_ok=True)
                y_ = np.array(results)
                with p_out.open('wb') as f:
                    pickle.dump(y_, f)
                log_msg = f'\nSaving logits at {str(p_out)}'
                continue

        return eval_results

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)
