from pathlib import Path
import csv
from itertools import chain
from glob import glob


p_filelist_dir = Path('/data/hyogun/repos/haawron_mmaction2/data/_filelists')
for p_filelist in map(Path, glob(f'{str(p_filelist_dir)}/**/*.txt', recursive=True)):
    if 'archive' in str(p_filelist):
        continue
    if 'test' not in p_filelist.stem or 'test_merged' in p_filelist.stem:
        continue
    valid_name = 'valid' if p_filelist.parent.name == 'ek100' else 'val'
    p_val_filelist = p_filelist.with_name(p_filelist.name.replace('test', valid_name))
    p_new_filelist = p_filelist.with_name(p_filelist.name.replace('test', 'test_merged'))
    with p_filelist.open() as f1, p_val_filelist.open() as f2, p_new_filelist.open('w') as f3:
        writer = csv.writer(f3, delimiter=' ')
        for line in chain(csv.reader(f1, delimiter=' '), csv.reader(f2, delimiter=' ')):
            writer.writerow(line)
    print(f'Done {p_new_filelist.name}')
