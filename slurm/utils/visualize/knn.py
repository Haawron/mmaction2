from slurm.utils.commons.patterns import get_full_infodict_from_jid
from slurm.utils.commons.videos import get_clip
from slurm.utils.commons.labelmaps import labelmaps
from mmcv import Config

import numpy as np

from pathlib import Path
from multiprocessing import Pool
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn)
import pickle
import csv
import shutil
import argparse
import time

from moviepy.editor import *


def main():
    parser = argparse.ArgumentParser(description='k-NN')
    parser.add_argument('-j', '--jid', type=str, default='5652_16', help='Slurm job id')  # 5652_16: osbp/tsm/ucf2hmdb, 5649_17: /osbp/tsm/hmdb2ucf
    parser.add_argument('-f', '--format', type=str, choices=['mp4', 'gif'], default='mp4', help='output file format')  # 5652_16: osbp/tsm/ucf2hmdb, 5649_17: /osbp/tsm/hmdb2ucf
    args = parser.parse_args()

    assert '_' in args.jid, 'Give the full JID'

    num_classes = 12
    labelmap = labelmaps['ucf']
    info = get_full_infodict_from_jid(args.jid)

    p_out_root = Path('visualizations/knn')
    p_out_root.mkdir(exist_ok=True, parents=True)

    source, target = info['source'], info['target']
    p_workdir = Path(info['ckpt']).parent
    p_target_test_pkl, = p_workdir.glob(f'features/{target}_test*')
    p_source_train_pkl, = p_workdir.glob(f'features/{source}_train*')
    p_target_train_pkl, = p_workdir.glob(f'features/{target}_train*')

    cfg = Config.fromfile(info['config'])

    features = []
    for p_pkl in [
        p_target_test_pkl,
        p_source_train_pkl,
        p_target_train_pkl
    ]:
        with p_pkl.open('rb') as f:
            feature_key = np.array(pickle.load(f))
            features.append(feature_key)

    p_ann_files = list(map(
        Path, [
            cfg['ann_file_test_target'],
            cfg['ann_file_train_source'],
            cfg['ann_file_train_target']
        ]
    ))

    prefixes = list(map(
        Path, [
            cfg['data_prefix_source'],
            cfg['data_prefix_target']
        ]
    ))

    anns = []
    for p_ann_file in p_ann_files:
        with p_ann_file.open('r') as f:
            reader = csv.reader(f, delimiter=' ')
            ann = np.array(list(reader))
            anns.append(ann)

    p_ann_file_source_open = p_ann_files[1].with_name(p_ann_files[1].name.replace('closed', 'open'))
    with p_ann_file_source_open.open('r') as f:
        reader = csv.reader(f, delimiter=' ')
        labels_source_open = np.array(list(reader))[:,-1].astype(np.int64)
        idx_source_train_closed = labels_source_open < num_classes
    features[1] = features[1][idx_source_train_closed]

    top5s = []  # list of np.array of strings(5, 3), [path, length, label]
    top5_preds = []  # list of np.array of ints(5)
    feature_query = features[0]
    for feature_key, ann in zip(features[1:], anns[1:]):
        score = feature_query @ feature_key.T
        idx_top5 = score.argsort(axis=1)[:,::-1][:,:5]  # top-5 key samples' indices (descending)
        top5 = ann[idx_top5,:]
        top5_pred = feature_key[idx_top5,:].argmax(axis=1)
        top5s.append(top5)
        top5_preds.append(top5_pred)

    # gathering jobs
    ann_query = anns[0]
    jobs = []
    count = 0
    for i, (top5, top5_pred) in enumerate(zip(top5s, top5_preds)):  # source_train or target_train
        p_outdir = p_out_root / info['model'] / info['backbone'] / info['dataset'] / ['source', 'target'][i]
        if p_outdir.is_dir():
            shutil.rmtree(p_outdir)
        p_outdir.mkdir(exist_ok=True, parents=True)
        for label, label_name in enumerate(labelmap):
            p_outdir_per_label = p_outdir / f'{label:02d}_{label_name}'
            p_outdir_per_label.mkdir()
        print('initialized outdir!', str(p_outdir))
        for row_query, row_keys, row_keys_pred_idx in zip(ann_query, top5, top5_pred):
            row_clips = []
            p_rawframe_query = prefixes[1] / row_query[0]
            p_rawframe_keys = list(map(lambda p: prefixes[i] / p, row_keys[:,0]))
            query_label = int(row_query[-1])
            row_clips.append((p_rawframe_query, labelmap[query_label], '', None))
            for p_rawframe_dir, key_label, key_pred in zip(
                p_rawframe_keys, map(int, row_keys[:,-1].tolist()), row_keys_pred_idx
            ):
                # clip_info = (p_rawframe_dir, labelmap[key_label], labelmap[key_pred], key_label==query_label)
                clip_info = (p_rawframe_dir, labelmap[key_label], '', key_label==query_label)
                row_clips.append(clip_info)
            job = p_outdir / f'{query_label:02d}_{labelmap[query_label]}' / f'{count:04d}.{args.format}', row_clips
            jobs.append(job)
            count += 1

    t_get_clip = 0
    t_appending_clips = 0
    t_wrinting_file = 0

    # running jobs
    with Pool() as pool, Progress(
        SpinnerColumn(),

        # *Progress.get_default_columns(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),  # not default
        TimeRemainingColumn(),

        MofNCompleteColumn()
    ) as progress:
        task = progress.add_task('[green]Processing...', total=len(jobs))
        for done, T in pool.imap_unordered(_worker, jobs):
            t_get_clip += T[0]
            t_appending_clips += T[1]
            t_wrinting_file += T[2]
            progress.update(task, advance=done)

    print(f'time elapsed for getting clips: {t_get_clip:.2f}s')
    print(f'time elapsed for appending clips: {t_appending_clips:.2f}s')
    print(f'time elapsed for wrinting files: {t_wrinting_file:.2f}s')


RED = (255, 0, 0)
GREEN = (0, 255, 0)


def _worker(job):
    p_out, row_clips = job
    clips = []
    for clip_info in row_clips:
        p_rawframe_dir, label_name, pred_name, is_correct = clip_info
        t0 = time.time()
        clip = get_clip(p_rawframe_dir, label_name, pred_name, 40, 10)
        t_get_clip = time.time() - t0
        if is_correct is not None:
            clip = clip.margin(5, color=GREEN if is_correct else RED).margin(1)
        clips.append(clip)
    t0 = time.time()
    final_clip = clips_array([clips]).resize(height=120)
    t1 = time.time()
    t_appending_clips = t1 - t0
    if p_out.suffix == '.mp4':
        final_clip.write_videofile(str(p_out), logger=None)
    elif p_out.suffix == '.gif':
        final_clip.write_gif(str(p_out), logger=None)
    t_wrinting_file = time.time() - t1
    return True, (t_get_clip, t_appending_clips, t_wrinting_file)


if __name__ == '__main__':
    main()
