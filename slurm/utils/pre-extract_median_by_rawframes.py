import numpy as np
from PIL import Image

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn)
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import os
from pathlib import Path
from functools import partial


progress_args = [
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),  # not default
    TimeRemainingColumn(),
    MofNCompleteColumn()
]


def main():
    p_out_root = Path(r'data/median')
    p_out_root.mkdir(exist_ok=True)
    print(f'Median Files Will Be Extracted in {str(p_out_root)}')
    print()

    p_all_videodirs = []
    for dataset in [
        # 'ucf101', 'hmdb51',
        # 'babel',
        'epic-kitchens-100'
    ]:
        print(f'\t{dataset}')
        print(f'\t\tInspecting the dataset has files not processed ...')
        p_dataset = Path(f'/local_datasets/{dataset}')
        print('\t\t\tListing leaf directories')
        # p_all_dirs = list(p for p in p_dataset.rglob('*') if p.is_dir())
        p_all_subdirs = [Path(p) for p in os.popen(f'find {str(p_dataset)} -type d').read().strip().split('\n')]
        print(f'\t\t\tAll subdirectories: {len(p_all_subdirs)}')
        print('\t\t\tDone')
        worker2 = partial(_worker_evaluating_if_existing, p_out_root)
        should_be_processeds = []
        with Pool() as pool, Progress(*progress_args) as progress:
            task = progress.add_task('\t\t\t[green]Evaluating directories if each should be processed...', total=len(p_all_subdirs))
            for should_be_processed in pool.imap_unordered(worker2, p_all_subdirs):
                should_be_processeds.append(should_be_processed)
                progress.update(task, advance=1)
        p_video_dirs = [p_dir for p_dir, should_be_processed in should_be_processeds if should_be_processed]
        print(f'\t\t{len(p_video_dirs)}')
        p_all_videodirs += p_video_dirs
        print(f'\t\tInspection of {dataset} done')

    worker = partial(_worker_extracting_median, p_out_root)
    with Pool(64) as pool, Progress(*progress_args) as progress:  # ~200G 필요
        task = progress.add_task('\t\t[green]Processing...', total=len(p_all_videodirs))
        for done in pool.imap_unordered(worker, p_all_videodirs):
            progress.update(task, advance=done)


def _worker_evaluating_if_existing(p_out_root, p_video_dir):
    p_out = p_out_root / Path(*p_video_dir.parts[2:]).with_suffix('.jpg')
    if p_out.is_file():
        should_be_processed = False
    else:
        if is_rawframe_dir(p_video_dir):
            should_be_processed = True
        else:
            should_be_processed = False
    return p_video_dir, should_be_processed


def _worker_extracting_median(p_out_root, p_video_dir):
    """
    {p_video_dir}/
    ├── frame_0000xxxx.jpg
    ├── ...
    └── frame_0000xxxx.jpg

    --> {p_out_root}/{*p_video_dir.parts[2:]}.jpg
    """
    p_frames = sorted(p_video_dir.glob('*.jpg'))
    if len(p_frames) > 100:
        indices = np.linspace(0, len(p_frames)-1, num=100, dtype=int)
        frames = [Image.open(p_frames[i]) for i in indices]
    else:
        frames = [Image.open(p_img) for p_img in p_frames]
    frame_array = np.array(list(map(np.array, frames)))
    median = np.median(frame_array, axis=0)
    median = Image.fromarray(median.astype(np.uint8))
    p_out = p_out_root / Path(*p_video_dir.parts[2:])
    p_out.parent.mkdir(parents=True, exist_ok=True)
    median.save(str(p_out.with_suffix('.jpg')))
    return True


def is_rawframe_dir(p_video_dir):
    result = False
    for _ in p_video_dir.glob('*.jpg'):  # if p contains jpg files
        result = True
        break
    return result


if __name__ == '__main__':
    main()
