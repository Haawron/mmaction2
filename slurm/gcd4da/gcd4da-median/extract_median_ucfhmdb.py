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

from pathlib import Path
from functools import partial


def main():
    p_out_root = Path(r'data/median')
    p_out_root.mkdir(exist_ok=True)

    p_videos = []
    for dataset in ['ucf101', 'hmdb51']:
        p = Path(rf'/local_datasets/{dataset}/rawframes')
        p_videos += list(p.glob('*/*'))  # depth = 2
        print(f'{dataset}: {len(p_videos)}')
    worker = partial(_worker, p_out_root)

    with Pool() as pool, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),  # not default
        TimeRemainingColumn(),
        MofNCompleteColumn()
    ) as progress:
        task = progress.add_task('[green]Processing...', total=len(p_videos))
        for done in pool.imap_unordered(worker, p_videos):
            progress.update(task, advance=done)


def _worker(p_out_root, p_video):
    p_frames = sorted(p_video.glob('*.jpg'))
    frames = [Image.open(p_img) for p_img in p_frames]
    frame_array = np.array(list(map(np.array, frames)))
    median = np.median(frame_array, axis=0)
    median = Image.fromarray(median.astype(np.uint8))
    p_out = p_out_root / Path(*p_video.parts[1:])
    p_out.parent.mkdir(parents=True, exist_ok=True)
    median.save(str(p_out.with_suffix('.jpg')))
    return True


if __name__ == '__main__':
    main()
