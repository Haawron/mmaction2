import csv
import io
import mmcv
from mmcv.fileio import FileClient
import decord
from typing import Union, Dict, List
from pathlib import Path
from tqdm import tqdm
import json
import re
from pprint import pprint


def can_be_loaded_by_decord(p_abs_video:Union[str,Path]):
    p_abs_video = str(p_abs_video)
    # file client가 필수인 것 같음
    file_client = FileClient(backend='disk')
    file_obj = io.BytesIO(file_client.get(p_abs_video))
    try:
        container = decord.VideoReader(file_obj, num_threads=1)
    except decord._ffi.base.DECORDError as e:
        can_be_load = False
    except RuntimeError as e:
        if 'Error reading' in str(e):
            can_be_load = False
        else:
            raise e
    else:
        del container
        can_be_load = True
    return can_be_load


def can_be_loaded_by_opencv(p_abs_video:Union[str,Path]):
    p_abs_video = str(p_abs_video)
    container = mmcv.VideoReader(p_abs_video)
    return True
    # try:
    #     container = mmcv.VideoReader(p_abs_video)
    # except Exception as e:
    #     # return False
    #     raise e
    # else:
    #     return True


def get_key_from_value(dict_of_list:Dict[str,List], value):
    for key, value_list in dict_of_list.items():
        if value in value_list:
            return key
    else:
        return None


p_k400_labelmap = Path('tools/data/kinetics/label_map_k400.txt')
with p_k400_labelmap.open('r') as f:
    k400_labelmap = [labelname.replace(' ', '_') for labelname, in csv.reader(f)]  # global index to labelname
k400_cdar_labelnames = {
    'jump': ['high_jump', 'long_jump'],
    'run': ['jogging', 'running_on_treadmill'],
    'throw': ['throwing_ball', 'javelin_throw', 'throwing_axe'],
    'kick': ['drop_kicking', 'kicking_field_goal', 'kicking_soccer_ball'],
    'bend': ['bending_back'],
    'dance': ['tap_dancing', 'swing_dancing', 'tango_dancing', 'dancing_macarena'],
    'clean_something': ['cleaning_windows', 'cleaning_floor', 'cleaning_pool', 'cleaning_toilet'],
    'squat': ['squat'],
    'punch': ['slapping', 'punching_bag', 'punching_person_(boxing)'],
    'crawl': ['crawling_baby'],
    'clap': ['clapping', 'applauding'],
    'pick_up': ['front_raises', 'deadlifting'],
    'unknown': [
        'driving_car', 'digging', 'climbing_a_rope', 'cutting_watermelon', 'golf_driving',
        'pushing_car', 'singing', 'unboxing', 'waxing_legs', 'smoking', 'sniffing', 'push_up',
        'kitesurfing', 'fixing_hair', 'balloon_blowing'
    ],
}
all_k400_cdar_labelnames = sum(map(list, k400_cdar_labelnames.values()), [])

have_we_downloaded_test_dataset = False  # False이면 val을 둘로 나눠서 val/test로 쓸 것임

# p_video_dir = Path('/data/dataset/kinetics400/videos')
p_video_dir = Path('/local_datasets/kinetics400/videos')
p_mmaction_split_dir = Path('tools/data/kinetics/kinetics400/annotations')
p_new_filelist_dir = Path('/data/hyogun/repos/haawron_mmaction2/data/_filelists/k400/processed')
splits = ['train', 'val', 'test']
mmaction_class_dict:Dict[str,List[List[str]]] = {}  # valid 한 애들의 class 별 모임의 split 별 모임
invalid_class_dict:Dict[str,List[List[str]]] = {}  # invalid 한 애들의 class 별 모임의 split 별 모임
for split in splits:
    p_new_filelists = [
        p_new_filelist_dir / f'filelist_k400_{split}_closed.txt',
        p_new_filelist_dir / f'filelist_k400_{split}_open.txt',
        p_new_filelist_dir / f'filelist_k400_{split}_open_all.txt',
    ]
    fs = [p.open('w') for p in p_new_filelists]
    writers = [csv.writer(f, delimiter=' ') for f in fs]

    split_of_mmaction = (
        split if have_we_downloaded_test_dataset
        else (
            'val' if split == 'test'
            else split
    ))
    p_mmaction_split = (p_mmaction_split_dir / ('kinetics_' + split_of_mmaction)).with_suffix('.csv')
    p_video_split_dir = p_video_dir / split_of_mmaction
    valid_data_per_class = [[] for _ in range(43)]
    invalid_data_per_class = [[] for _ in range(43)]
    valid_annotation = []
    invalid_annotation = []
    print()
    with p_mmaction_split.open('r') as f:
        reader = csv.reader(f)
        header = next(reader)
        reader = list(reader)
        counts = {
            'our setting': [0, {
                'file exists': [0, {
                    'can be loaded': 0,
                    'cannot be loaded': 0}],
                'file not exists': 0}],
            'not our setting': 0
        }
        for i, line in enumerate(reader):
            if split in ['val', 'test']:
                if not have_we_downloaded_test_dataset and i % 2 == (1 if split == 'val' else 0):
                    # testset이 없으면 둘 중 val을 반반 나눔
                    # 정확히 나누려면 코드 좀 많이 고쳐야 되니까 홀짝으로 나눔
                    continue
            labelname, youtube_id, t_start, t_end, _ = line
            p_video = p_video_split_dir / labelname / f'{youtube_id}_{int(t_start):06d}_{int(t_end):06d}.mp4'
            labelname = labelname.replace(' ', '_')
            # is_our_setting = labelname in k400_cdar_labelnames
            parent_name = get_key_from_value(k400_cdar_labelnames, labelname)
            is_our_setting = parent_name is not None
            if is_our_setting:
                idx = list(k400_cdar_labelnames.keys()).index(parent_name)
                is_file = p_video.is_file()
                can_be_loaded = is_file and can_be_loaded_by_decord(p_video)
                if is_file and can_be_loaded:
                    valid_annotation.append(line)
                    line = [f'{labelname}/{p_video.stem}', 30*(int(t_end)-int(t_start)), idx]
                    if idx < 12:  # closed
                        writers[0].writerow(line)
                    writers[1].writerow(line)  # open: unknowns included
                    if idx == 12:  # open_all: unknowns should be discriminated
                        global_idx = 12 + k400_cdar_labelnames['unknown'].index(labelname)
                        writers[2].writerow(line[:-1]+[global_idx])
                    else:
                        writers[2].writerow(line)
                else:
                    print(idx, is_file, can_be_loaded, str(p_video).replace('local_datasets', 'data/dataset').replace(' ', '_'))
                    invalid_annotation.append(line)

            if is_our_setting:
                counts['our setting'][0] += 1
                if is_file:
                    counts['our setting'][1]['file exists'][0] += 1
                    if can_be_loaded:
                        counts['our setting'][1]['file exists'][1]['can be loaded'] += 1
                    else:
                        counts['our setting'][1]['file exists'][1]['cannot be loaded'] += 1
                else:
                    counts['our setting'][1]['file not exists'] += 1
            else:
                counts['not our setting'] += 1
            print(f'\r[{i+1}/{len(reader)}] ', end='')
            if i % (len(reader)//4) == 1:
                print()
                pprint(counts)
                print()

    for labelname, youtube_id, t_start, t_end, _ in valid_annotation:
        valid_data_per_class[all_k400_cdar_labelnames.index(labelname.replace(' ', '_'))].append(youtube_id)
    for labelname, youtube_id, _, _, _ in invalid_annotation:
        invalid_data_per_class[all_k400_cdar_labelnames.index(labelname.replace(' ', '_'))].append(youtube_id)

    mmaction_class_dict[split] = valid_data_per_class
    invalid_class_dict[split] = invalid_data_per_class

    for f in fs: f.close()

p_states_dir = Path('tools/data/kinetics/kinetics400/states')
p_states_dir.mkdir(exist_ok=True, parents=True)
for state, data in zip(['success', 'failed'], [mmaction_class_dict, invalid_class_dict]):
    p_json = p_states_dir / f'{state}.json'
    with p_json.open('w') as f:
        s = json.dumps(data, indent=2)
        s = re.sub(r'\[\n\s*', '[', s)
        s = re.sub(r'\[(["\w-]+,?)\n\s+', r'[\1', s)
        s = re.sub(r'\n\s+\]', r']', s)
        s = re.sub(r'("[\w-]+",)\n\s+', r'\1', s)
        f.write(s)
