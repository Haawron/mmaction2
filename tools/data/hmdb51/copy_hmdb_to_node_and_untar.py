import os
from pathlib import Path
from shutil import copy
import tarfile
import argparse


parser = argparse.ArgumentParser(description='Copy And Untar EK Files to Local')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='')
global_args = parser.parse_args()


def print_wrap(*args, **kwargs):
    if global_args.verbose:
        print(*args, **kwargs)


# copying to computing node not master
assert os.uname().nodename != 'master'
p_root = Path(r'/data/dataset/epic-kitchens-100')
p_dst_root = Path(r'/local_datasets/epic-kitchens-100')

# copy tar files
participants = ['P22', 'P02', 'P04']
for participant in participants:
    print_wrap('\n\n', participant)
    p_rel = f'EPIC-KITCHENS/{participant}/rgb_frames'
    p_src = p_root / p_rel
    p_dst = p_dst_root / p_rel
    p_dst.mkdir(exist_ok=True, parents=True)
    for p_file in p_src.glob('*.tar'):
        print_wrap(p_file, '-->', p_dst / p_file.name)
        if (p_dst / p_file.name).is_file():
            print_wrap('pass')
            continue
        copy(p_file, p_dst)

# untar
for p_tar in p_dst_root.glob('**/*.tar'):
    p_target_dir = p_tar.with_suffix('')
    print_wrap(f'Untar', p_tar, '\tto\t', p_target_dir)
    if p_tar.with_suffix('').is_dir():
        print_wrap('pass')
        continue
    with tarfile.open(p_tar) as f_tarfile:
        try:
            f_tarfile.extractall(p_target_dir)
        except tarfile.ReadError:
            print_wrap(p_tar, 'is corrupted!')

# del?