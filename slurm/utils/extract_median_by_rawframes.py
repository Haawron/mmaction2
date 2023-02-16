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

from pathlib import Path
from functools import partial


# 메모리 200GB 이상 필요
def main():
    p_out_root = Path(r'data/median')
    p_out_root.mkdir(exist_ok=True)

    p_all_videodirs = []
    for dataset in [
        'ucf101', 'hmdb51',
        'babel'
    ]:
        print(f'{dataset} ...')
        p_dataset = Path(f'/local_datasets/{dataset}')
        p_all_dirs = list(p_dataset.rglob('*'))
        worker2 = partial(_worker2, p_out_root)
        should_be_processeds = []
        with ThreadPool() as pool:
            for should_be_processed in pool.imap(worker2, p_all_dirs):
                should_be_processeds.append(should_be_processed)
        p_video_dirs = [
            p_dir
            for p_dir, should_be_processed in zip(p_all_dirs, should_be_processeds)
            if should_be_processed
        ]
        print(f'\t{len(p_video_dirs)}')
        p_all_videodirs += p_video_dirs

    worker = partial(_worker, p_out_root)
    with Pool(16) as pool, Progress(  # ~200G 필요
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),  # not default
        TimeRemainingColumn(),
        MofNCompleteColumn()
    ) as progress:
        task = progress.add_task('[green]Processing...', total=len(p_all_videodirs))
        for done in pool.imap_unordered(worker, p_all_videodirs):
            progress.update(task, advance=done)


def _worker(p_out_root, p_video_dir):
    p_frames = sorted(p_video_dir.glob('*.jpg'))
    frames = [Image.open(p_img) for p_img in p_frames]
    frame_array = np.array(list(map(np.array, frames)))
    median = np.median(frame_array, axis=0)
    median = Image.fromarray(median.astype(np.uint8))
    p_out = p_out_root / Path(*p_video_dir.parts[2:])
    p_out.parent.mkdir(parents=True, exist_ok=True)
    median.save(str(p_out.with_suffix('.jpg')))
    return True


def _worker2(p_out_root, p_video_dir):
    return should_be_processed(p_out_root, p_video_dir)


def should_be_processed(p_out_root, p_video_dir):
    p_out = p_out_root / Path(*p_video_dir.parts[2:]).with_suffix('.jpg')
    if p_out.is_file():
        return False
    if is_rawframe_dir(p_video_dir):
        return True
    return False


def is_rawframe_dir(p_video_dir):
    result = False
    for _ in p_video_dir.glob('*.jpg'):  # if p contains jpg files
        result = True
        break
    return result


if __name__ == '__main__':
    main()
