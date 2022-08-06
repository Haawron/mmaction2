import numpy as np

from multiprocessing import Pool
from pathlib import Path
from rich.progress import Progress, MofNCompleteColumn
from functools import partial

from commons.patterns import get_infodict_and_pattern_from_workdir_name

import csv
import re
import shutil
import argparse

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def main():
    p_prediction = Path(r'work_dirs/train_output/ek100/tsm/vanilla/P02/source-only/16847__vanilla_tsm_P02_source-only/0/20220402-053842/predictions/osvm.csv')
    _, pattern = get_infodict_and_pattern_from_workdir_name(p_prediction)
    info = re.match(pattern, str(p_prediction)).groupdict()
    dataset = info['dataset']
    prefix = Path(r'/local_datasets/epic-kitchens-100/EPIC-KITCHENS')

    assert Path(r'/local_datasets').is_dir(), 'Do not run this script on master'

    labelmap = {
        'ek100': [
            'take',
            'put',
            'wash',
            'open',
            'close',
            'unk',
        ]
    }[dataset]

    with p_prediction.open('r') as f:
        reader = csv.reader(f, delimiter=' ')
        if dataset == 'ek100':
            data = np.array(list(reader))
        else:
            pass
            # paths, _, labels, preds = map(np.array, zip(*reader))

    p_outdir = Path(r'failure_modes_visualized')
    if p_outdir.exists():
        shutil.rmtree(p_outdir)
    p_outdir.mkdir()
    for label_name in labelmap:
        (p_outdir / label_name).mkdir()

    wrong = data[:,-1] != data[:,-2]  # preds != labels
    all_jobs = [(row[0], *map(int, row[1:])) for row in data[wrong]]
    num_labels = max([job[-2] for job in all_jobs])
    counts = [0] * (num_labels + 1)
    jobs = []
    for job in all_jobs:
        pred = job[-1]
        counts[pred] += 1
        if counts[pred] <= 30:
            jobs.append(job)
    with Pool() as pool, Progress(*Progress.get_default_columns(), MofNCompleteColumn()) as progress:
        task = progress.add_task('[green]Processing...', total=len(jobs))
        for done in pool.imap_unordered(
            partial(worker, labelmap=labelmap, info=info, prefix=prefix, p_outdir=p_outdir),
            jobs
        ):
            progress.update(task, advance=done)


def worker(row, labelmap, info, prefix, p_outdir):
    if info['dataset'] == 'ek100':
        path, start, length, label, pred = row
    else: 
        pass
    pred_name = labelmap[pred]
    label_name = labelmap[label]
    p_images = list(map(str, sorted((prefix / path).glob('*.jpg'))[start-1:start-1+length]))
    p_out = p_outdir / '/'.join(info.values()) / f'{pred}_{pred_name}' / f'{pred_name}_{label_name}_{path.split("/")[-1]}.mp4'
    if p_out.is_file():
        p_out = p_out.with_name(p_out.stem + 'o.mp4')
    clip = ImageSequenceClip(p_images, fps=30.)
    # clip.write_videofile(str(p_out), logger=None)
    clip.write_gif(str(p_out.with_suffix('.gif')), logger=None)
    return True


if __name__ == '__main__':
    main()
