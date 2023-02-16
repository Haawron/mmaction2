import decord
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

import os
from pathlib import Path

from mmaction.datasets.pipelines.loading import DecordInit, DecordDecode


# 메모리 200GB 이상 필요
def main():
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    print(f'# cpus: {n_cpus}')

    p_out_root = Path('data/median')
    p_out_root.mkdir(exist_ok=True)

    p_all_mp4s = []
    p_all_outs = []
    for sub_path, domain_name in [
        ['kinetics400/videos', 'k400']
    ]:
        print(f'{domain_name} ...')
        # /local_datasets/kinetics400/videos/train/abseiling/0347ZoDXyP0_000095_000105.mp4
        # => train/abseiling/0347ZoDXyP0_000095_000105.mp4
        # => p_out = data/median/k400/train/abseiling/0347ZoDXyP0_000095_000105.jpg
        # check if already exists
        p_dataset = Path(f'/local_datasets/{sub_path}')
        p_mp4s = list(p_dataset.rglob('*.mp4'))
        n_mp4s = len(p_mp4s)

        p_outdir = p_out_root / domain_name
        p_tbp_mp4s = []
        p_outs = []
        for p_mp4 in p_mp4s:
            p_out = (p_outdir / Path(*p_mp4.parts[list(p_mp4.parts).index(sub_path.split('/')[-1])+1:])).with_suffix('.jpg')
            if not p_out.is_file():
                p_tbp_mp4s.append(p_mp4)
                p_outs.append(p_out)
        n_tbp_mp4s = len(p_tbp_mp4s)
        print(f'{domain_name}: [{n_tbp_mp4s:6d}/{n_mp4s:6d}] [# to be processed/ # all]')
        p_all_mp4s += p_tbp_mp4s
        p_all_outs += p_outs

    assert len(p_all_mp4s) == len(p_all_outs)

    with Pool(n_cpus) as pool, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),  # not default
        TimeRemainingColumn(),
        MofNCompleteColumn()
    ) as progress:
        task = progress.add_task('[green]Processing...', total=len(p_all_mp4s))
        for done in pool.imap_unordered(worker, zip(p_all_mp4s, p_all_outs), chunksize=10):
            progress.update(task, advance=done)


def worker(args):
    p_mp4, p_out = args
    try:
        results = DecordInit()({'filename': p_mp4})
    except (decord._ffi.base.DECORDError, RuntimeError) as e:
        return True
    results['frame_inds'] = np.arange(results['total_frames'])
    results = DecordDecode(mode='efficient')(results)
    x = np.array(results['imgs'])  # [T, H, W, C]
    median = np.median(x, axis=0)  # [H, W, C]
    median = Image.fromarray(median.astype(np.uint8))
    p_out.parent.mkdir(parents=True, exist_ok=True)
    median.save(str(p_out))
    return True


if __name__ == '__main__':
    main()
