import torch
import torch.nn as nn
import numpy as np

from mmaction.apis import init_recognizer
from mmaction.datasets import build_dataset, build_dataloader
from mmaction.core import OutputHook

from sklearn import svm

import os
import psutil
import argparse


# python slurm/osvm/extract_features.py -j 16279


def get_whole_dataset(dataloader, layer='cls_head', verbose=True):
    svm_inputs = {'scores': [], 'features': [], 'labels': []}
    for i, batch in enumerate(dataloader):
        if i % 10 == 9 and verbose:
            print(f'{i+1:4d}/{len(dataloader)}...', end=' ')
            print(f'Currently occupying {psutil.Process(os.getpid()).memory_info().rss / 1024**3:.3f} GiB')
        imgs, labels = batch['imgs'], batch['label']
        scores, features = get_scores_and_features(model, imgs, layer)
        svm_inputs['scores'].append(scores)
        svm_inputs['features'].append(features.cpu().numpy())
        svm_inputs['labels'].append(labels)
    if i > 0:  # for debug
        svm_inputs['scores'] = np.concatenate(svm_inputs['scores'])
        svm_inputs['features'] = np.concatenate(svm_inputs['features'])
        svm_inputs['labels'] = np.concatenate(svm_inputs['labels'])
    else:
        svm_inputs['scores'] = svm_inputs['scores'][0]
        svm_inputs['features'] = svm_inputs['features'][0]
        svm_inputs['labels'] = svm_inputs['labels'][0]
    if verbose:
        print('done')
    return svm_inputs


def get_scores_and_features(model, x, layer='cls_head'):
    outputs = [layer]
    with OutputHook(model, outputs=outputs, as_tensor=True) as h:
        with torch.no_grad():
            scores = model(x.cuda(), return_loss=False)[0]
        returned_features = h.layer_outputs if outputs else None
    feature = returned_features[layer]
    if layer == 'backbone':
        feature = gap_and_consensus(model, feature)
    return scores, feature


def gap_and_consensus(model, backbone_output):
    if model.cls_head.is_shift and model.cls_head.temporal_pool:
        # [2 * N, num_segs // 2, num_classes]
        feature = backbone_output.view((-1, model.cls_head.num_segments // 2) + backbone_output.size()[1:])
    else:
        # [N, num_segs, num_classes]
        feature = backbone_output.view((-1, model.cls_head.num_segments) + backbone_output.size()[1:])
    consensus = nn.AdaptiveAvgPool2d(1)(feature.mean(dim=1)).squeeze() 
    return consensus


parser = argparse.ArgumentParser(description='Print the Best Model\'s Info, Given JID Or Config Vars.')
parser.add_argument('-j', '--jid', default=None,
                    help='')
args = parser.parse_args()


line = os.popen(f'python slurm/print_best_scores.py -j {args.jid} -o').read().split()
checkpoint, config = line[-2:]

print(checkpoint, config)
model = init_recognizer(
    config=config,
    checkpoint=checkpoint,
    device='cuda'
)
cfg = model.cfg
num_classes = cfg.num_classes

dataset_setting = cfg.data.train
dataloader_setting = dict(
    videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    persistent_workers=cfg.data.get('persistent_workers', False),
    num_gpus=1,
    dist=False,
    seed=999,
    **cfg.data.get('test_dataloader', {})
)
dataset = build_dataset(dataset_setting)
dataloader = build_dataloader(dataset, **dataloader_setting)
features = get_whole_dataset(dataloader)
print('saving npz ...')
np.savez('slurm/osvm/features/features.npz', **features)
print('done')
