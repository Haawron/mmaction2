import numpy as np
import pandas as pd
from PIL import Image

from typing import List

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
    p_out_root = Path(r'data/median/ek100-mm-sada-split')
    p_out_root.mkdir(exist_ok=True)
    print(f'Median Files Will Be Extracted in {str(p_out_root)}')
    print()

    print(f'====== EK 100 ======')
    # print(f'Inspecting the dataset has files not processed ...')
    p_all_videos_in_rawframes:List[List[Path]] = []
    p_rawframe_root = Path(f'/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb')

    def aggregator(row):
        p_video = p_rawframe_root / row['path']
        indices = np.char.array(np.arange(row['start_frame'], row['start_frame']+row['length']+1), unicode=True)
        p_frames = str(p_video) + '/frame_' + indices.zfill(10) + '.jpg'
        return list(map(Path, p_frames.tolist()))

    header = ['path', 'start_frame', 'length', 'label']
    for i in range(1, 4):
        df_filelist = pd.read_csv(f'data/_filelists/ek100/mm-sada_resampled/D{i}_train.txt', header=None, delimiter=' ', names=header)
        df_filelist['p_frames'] = df_filelist.agg(aggregator, axis=1)
        p_all_videos_in_rawframes += df_filelist['p_frames'].tolist()  # list of lists + list of lists of frames

    worker = partial(_worker_extracting_median, p_out_root)
    with Pool() as pool, Progress(*progress_args) as progress:  # ~200G 필요
        task = progress.add_task('\t\t[green]Processing...', total=len(p_all_videos_in_rawframes))
        for done in pool.imap_unordered(worker, p_all_videos_in_rawframes):
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


def _worker_extracting_median(p_out_root, p_frames:List[Path]):
    """
    {p_video_dir}/
    ├── frame_0000xxxx.jpg
    ├── ...
    └── frame_0000xxxx.jpg

    --> {p_out_root}/{*p_video_dir.parts[2:]}.jpg
    """
    if len(p_frames) > 100:
        indices = np.linspace(0, len(p_frames)-1, num=100, dtype=int)
        frames = [Image.open(p_frames[i]) for i in indices]
    else:
        frames = [Image.open(p_img) for p_img in p_frames]
    frame_array = np.array(list(map(np.array, frames)))
    median = np.median(frame_array, axis=0)
    median = Image.fromarray(median.astype(np.uint8))
    *_, split, domain, video_id, _ = p_frames[0].parts
    p_start, *_, p_end = sorted(p_frames)
    i_start, i_end = p_start.stem.split('_')[1], p_end.stem.split('_')[1]
    p_out = p_out_root / split / domain / video_id / f'{i_start}_{i_end}.jpg'
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
