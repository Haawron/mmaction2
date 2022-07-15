from pathlib import Path
import numpy as np
import re
import csv
import imageio
import shutil
import argparse
from multiprocessing import Pool

import os
os.chdir('/data/hyogun/repos/haawron_mmaction2')


def main():
    """
    Checklist:
        - 
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-p', '--p-test',
        default=Path(r'work_dirs/train_output/ucf-hmdb/tsm/gcd4da/phase0/full/ucf-hmdb/20146__GCD4DA-phase0-all_tsm_ucf-hmdb/3/20220518-162226/filelist_test_hmdb_k016.txt'),
        type=Path, help='')
    args = parser.parse_args()
    
    assert args.p_test.is_file(), f'wrong path'

    print(args.p_test.stem)
    (domain, k), = re.findall(r'test_(\w+)_k(\d+)', args.p_test.stem)
    k = int(k)
    print(f'Domain: {domain}, k: {k}')

    with args.p_test.open('r') as f:
        preds = np.array([int(line[-1]) for line in csv.reader(f, delimiter=' ')])

    domain2dataset = {
        'ucf': 'ucf101',
        'hmdb': 'hmdb51',
        'P02': 'epic-kitchens-100',
        'P04': 'epic-kitchens-100',
        'P22': 'epic-kitchens-100',
    }
    p_dir_filelist = Path(r'data/_filelists')
    p_ann = p_dir_filelist / domain2dataset[domain] / f'filelist_{domain}_test_open.txt'
    with p_ann.open('r') as f:
        clips = np.array(list(csv.reader(f, delimiter=' ')))
        labels = np.array([int(line[-1]) for line in clips])

    num_classes = (
        5 if 'P' in domain else
        12 if domain in ['ucf', 'hmdb'] else
        None
    )
    idx_to_label = (
        ['take', 'put', 'wash', 'open', 'close'] if 'P' in domain else
        ['climb', 'fencing', 'golf', 'kick-ball', 'pullup', 'punch', 'pushup', 'ride-bike', 'ride-horse', 'shoot-ball', 'shoot-bow', 'walk'] if domain in ['ucf', 'hmdb'] else
        None
    )
    idx_to_label.append('unknown')
    correct = np.minimum(preds, num_classes) == labels
    num_samples = 5
    args_correct, args_incorrect = {}, {}
    for c in range(k):
        args_correct[c]   = ((preds==c) & correct).nonzero()[0][:num_samples]
        args_incorrect[c] = ((preds==c) & ~correct).nonzero()[0][:num_samples]

    print()
    print(f'# samples / class: {num_samples}')
    print(f'{"k":2s}' + ' '*5 + f'{"correct args":25s} incorrect args')
    for key in args_correct.keys():
        print(f'{key:2d}' + ' '*5 + f'{str(args_correct[key]):25s} {str(args_incorrect[key])}')
    
    p_base_video_dir = Path(rf'/local_datasets/{domain2dataset[domain]}/rawframes')
    p_out_dir_base = args.p_test.parent / 'failure_modes'
    max_gif_length = 200

    if p_out_dir_base.is_dir():
        shutil.rmtree(p_out_dir_base)

    filename_tmpl = (
        'frame_{:010d}.jpg' if 'P' in domain else
        'img_{:05d}.jpg' if domain in ['ucf', 'hmdb'] else
        None
    )

    jobs = []
    for correctness, args_ in zip(['o', 'x'], [args_correct, args_incorrect]):
        p_out_dir = p_out_dir_base / correctness
        if not p_out_dir.is_dir():
            p_out_dir.mkdir(parents=True, exist_ok=True)
        for c in range(k):
            for i, test_indices in enumerate(args_[c]):
                line = clips[test_indices]
                if len(line) == 3:
                    rel_p, length, label = line
                    start = 1
                elif len(line) == 4:
                    rel_p, start, length, label = line
                    start = int(start)
                else:
                    assert False, f'invalid filelist: # elems = {len(line)} ∉ (3, 4)'
                
                length = int(length)
                label = int(label)
                p_video = p_base_video_dir / rel_p

                jobs.append({
                    'cluster_idx': c,
                    'gif_idx': i,
                    'p_video': p_video,
                    'p_out_dir': p_out_dir,
                    'pred_label_name': idx_to_label[min(num_classes, c)],
                    'gt_label_name': idx_to_label[min(num_classes, label)],
                    'start': start,
                    'length': length,
                    'max_gif_length': max_gif_length,
                    'filename_tmpl': filename_tmpl,
                })

    with Pool() as p:
        count = 0
        for ret in p.imap_unordered(worker, jobs):
            count += ret
            print(f'{count:3d}/{len(jobs):3d}')
    p.join()


def worker(job:dict):
    write_gif(**job)
    return 1


def write_gif(
    cluster_idx:int, gif_idx:int,
    p_video:Path, p_out_dir:Path, 
    pred_label_name:str, gt_label_name:str,
    start:int, length:int, max_gif_length:int,
    filename_tmpl:str):
    
    with imageio.get_writer(
        p_out_dir / f'k{cluster_idx:02d}_{gif_idx:02d}_{pred_label_name}_{gt_label_name}.gif',
        mode='I', duration=1/30) as writer:
        last_frame_idx = start + min(max_gif_length, length) - 1
        for frame_idx in range(start, last_frame_idx+1):
            p_frame = p_video / filename_tmpl.format(frame_idx)
            image = imageio.imread(p_frame)
            writer.append_data(image)
    

if __name__ == '__main__':
    main()
