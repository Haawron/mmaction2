from pathlib import Path
import csv
from math import inf
import torch


p_root = Path('/data/hyogun/repos/haawron_mmaction2/data/epic-kitchens-100/EPIC-KITCHENS')
p_annotation = Path(f'/data/hyogun/repos/epic-kitchens-100-annotations/EPIC_100_train.csv')
p_filelist_dir = Path('/data/hyogun/repos/haawron_mmaction2/data/epic-kitchens-100')  # output path

def writerow(writer, video_info):
    p_video = Path() / video_info['participant_id'] / 'rgb_frames' / video_info['video_id']
    start = int(video_info['start_frame'])
    end = int(video_info['stop_frame'])
    total_frames = end - start + 1
    label = video_info['verb_class']
    writer.writerow([p_video, start, total_frames, label])


max_count = inf
participants = ['P22', 'P02', 'P04']
# missed = ['P22_10', 'P02_103', 'P04_06', 'P04_30']
missed = ['P04_30']
split_names = ['train', 'valid', 'test']
labelset_names = ['open', 'closed']
weights = [.7, .2, .1]
with p_annotation.open('r', newline='') as f_annotation:
    reader = csv.reader(f_annotation)
    header = next(reader)
    all_videos = {participant: [] for participant in participants}
    for video in reader:
        video = {key: value for key, value in zip(header, video)}
        participant = video['participant_id']
        if participant in participants and video['video_id'] not in missed:  # take put wash open close
            all_videos[participant].append(video)
    for participant in participants:
        videos = all_videos[participant]
        lengths = [int(len(videos) * weight) for weight in weights]
        lengths[0] += len(videos) - sum(lengths)
        splits = torch.utils.data.random_split(videos, lengths, generator=torch.Generator().manual_seed(999))
        for split, split_name in zip(splits, split_names):
            for labelset_name in labelset_names:
                p_filelist = p_filelist_dir / f'filelist_{participant}_{split_name}_{labelset_name}.txt'
                count = 0
                with p_filelist.open('w', newline='') as f_filelist:
                    writer = csv.writer(f_filelist, delimiter=' ')
                    for i, video in enumerate(split):
                        if video['verb_class'] not in list('01234'):
                            if labelset_name == 'open':
                                video['verb_class'] = '5'
                            else:
                                continue
                        writerow(writer, video)
                        count += 1
                        if count >= max_count:
                            break
