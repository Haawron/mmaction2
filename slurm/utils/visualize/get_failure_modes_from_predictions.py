import numpy as np

from multiprocessing import Pool
from pathlib import Path
from rich.progress import Progress, MofNCompleteColumn
from functools import partial

from commons.patterns import (
    get_full_infodict_from_jid,
    get_p_target_workdir_name_with_jid
)

import csv
import re
import shutil
import argparse

from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# from mmcv import Config
# cfg = Config.fromfile(config)


def main():
    parser = argparse.ArgumentParser(description='Failure Modes')
    parser.add_argument('-j', '--jid', type=str, default='16847_0', help='Slurm job id')
    parser.add_argument('-f', '--format', type=str, choices=['mp4', 'gif'], default='mp4', help='Output format')
    parser.add_argument('-m', '--matrix', action='store_true',
                        help='Whether to form output as a matrix')
    parser.add_argument('--osvm', action='store_true',
                        help='Whether to use osvm result which is in ${expdir}/predictions/osvm.csv otherwise use ${expdir}/best_pred.pkl')
    parser.add_argument('--target-test', action='store_true',  # todo: ek100은 source-only 하나 당 target이 두 개임
                        help='Whether to use target test result which is in ${expdir}/predictions/target-test.csv. This option is only for vanilla')
    args = parser.parse_args()

    assert '_' in args.jid, 'Give the full JID'
    # p_prediction = Path(r'work_dirs/train_output/ek100/tsm/vanilla/P02/source-only/16847__vanilla_tsm_P02_source-only/0/20220402-053842/predictions/osvm.csv')
    p_workdir = get_p_target_workdir_name_with_jid(args.jid)
    p_prediction = p_workdir / (
        'predictions/osvm.csv' if args.osvm else
        'predictions/target-test.csv' if args.target_test else
        'best_pred.pkl'
    )

    info = get_full_infodict_from_jid(args.jid)
    print(info)
    dataset = info['dataset']
    target = info['target']
    domain2name = {'ucf': 'ucf101', 'hmdb': 'hmdb51', 'P02': 'ek100', 'P04': 'ek100', 'P22': 'ek100'}
    if dataset == 'ek100':
        data_prefix = Path(r'/local_datasets/epic-kitchens-100/EPIC-KITCHENS')
    else:
        data_prefix = Path(rf'/local_datasets/{domain2name[target]}/rawframes')
    p_ann_file = Path(rf'data/_filelists/{domain2name[target]}/filelist_{target}_test_open.txt')

    assert Path(r'/local_datasets').is_dir(), 'Do not run this script on master'
    assert data_prefix.is_dir(), 'Dataset not available'

    labelmap = {
        'ek100': [
            'take',
            'put',
            'wash',
            'open',
            'close',
            'unk',
        ],
        'ucfhmdb': [
            'climb',            # 0
            'fencing',          # 1
            'golf',             # 2
            'kick-ball',        # 3
            'pullup',           # 4

            'punch',            # 5
            'pushup',           # 6
            'ride-bike',        # 7
            'ride-horse',       # 8
            'shoot-ball',       # 9

            'shoot-bow',        # 10
            'walk',             # 11
            'unk',
        ],
    }['ucfhmdb' if dataset in ['ucf', 'hmdb'] else dataset]

    with p_prediction.open('r') as f:
        reader = csv.reader(f, delimiter=' ')
        data = np.array(list(reader))

    p_outdir = Path(r'failure_modes_visualized')
    if p_outdir.exists():
        shutil.rmtree(p_outdir)
    p_outdir.mkdir()
    for label_name in labelmap:
        (p_outdir / label_name).mkdir()

    wrong = data[:,-1] != data[:,-2]  # preds != labels
    all_jobs = [(row[0], *map(int, row[1:])) for row in data[wrong]]  # failed samples' filelist row
    num_labels = max([job[-2] for job in all_jobs])
    counts = [0] * (num_labels + 1)
    jobs = []
    for job in all_jobs:
        pred = job[-1]
        counts[pred] += 1
        if counts[pred] <= 16:
            jobs.append(job)

    if args.matrix:
        worker = partial(
                _worker,
                labelmap=labelmap,
                dataset=dataset,
                jid=args.jid,
                prefix=data_prefix,
                p_outdir=p_outdir,
                extension=args.format,
            )
        for clip in pool.imap_unordered(_worker_for_matrix, jobs):
            pass
    else:
        with Pool() as pool, Progress(*Progress.get_default_columns(), MofNCompleteColumn()) as progress:
            task = progress.add_task('[green]Processing...', total=len(jobs))
            worker = partial(
                _worker,
                labelmap=labelmap,
                dataset=dataset,
                jid=args.jid,
                prefix=data_prefix,
                p_outdir=p_outdir,
                extension=args.format,
            )
            for done in pool.imap_unordered(worker, jobs):
                progress.update(task, advance=done)


def _worker(row, labelmap, dataset, jid, prefix, p_outdir, extension):
    # if dataset == 'ek100':
    #     path, start, length, label, pred = row
    # else:
    #     path, length, label, pred = row
    #     start = 0
    # pred_name = labelmap[pred]
    # label_name = labelmap[label]
    # p_images = list(map(str, sorted((prefix / path).glob('*.jpg'))[start-1:start-1+length]))
    # p_out = p_outdir / jid / f'{pred}_{pred_name}' / f'{pred_name}_{label_name}_{path.split("/")[-1]}.{extension}'
    # if p_out.is_file():
    #     p_out = p_out.with_name(p_out.stem + f'o.{extension}')
    # clip = ImageSequenceClip(p_images, fps=30.)

    clip, p_out = _get_clip(row, labelmap, dataset, jid, prefix, p_outdir, extension, True)

    if extension == 'mp4':
        clip.write_videofile(p_out, logger=None)
    elif extension == 'gif':
        clip.write_gif(p_out, logger=None)
    return True


def _worker_for_matrix(row, labelmap, dataset, jid, prefix, p_outdir, extension):
    clip, pred, label = _get_clip(row, labelmap, dataset, jid, prefix, p_outdir, extension, False)
    txt_clip = TextClip("GeeksforGeeks", fontsize = 75, color = 'black')

    # setting position of text in the center and duration will be 10 seconds
    txt_clip = txt_clip.set_pos('center').set_duration(10)

    # Overlay the text clip on the first video clip
    video = CompositeVideoClip([clip, txt_clip])
    return clip, pred


def _get_clip(row, labelmap, dataset, jid, prefix, p_outdir, extension, return_path:bool):
    if dataset == 'ek100':
        path, start, length, label, pred = row
    else:
        path, length, label, pred = row
        start = 0
    p_images = list(map(str, sorted((prefix / path).glob('*.jpg'))[start-1:start-1+length]))
    clip = ImageSequenceClip(p_images, fps=30.)

    if return_path:
        pred_name = labelmap[pred]
        label_name = labelmap[label]
        p_out = p_outdir / jid / f'{pred}_{pred_name}' / f'{pred_name}_{label_name}_{path.split("/")[-1]}.{extension}'
        if p_out.is_file():
            p_out = p_out.with_name(p_out.stem + f'o.{extension}')
        p_out = str(p_out)
        return clip, p_out
    else:
        return clip, pred, label  # ImageSequenceClip, int, int


if __name__ == '__main__':
    main()
