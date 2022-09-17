from pathlib import Path
import pickle
import numpy as np

import csv
import shutil

from mmcv import Config
from mmaction.core import confusion_matrix

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import *


dataset = 'ucf'
labelmap = {
    'ek100': [
        'take',
        'put',
        'wash',
        'open',
        'close',
        'unk',
    ],
    'ucfhmdb': [
        'climb',            # 0
        'fencing',          # 1
        'golf',             # 2
        'kick-ball',        # 3
        'pullup',           # 4

        'punch',            # 5
        'pushup',           # 6
        'ride-bike',        # 7
        'ride-horse',       # 8
        'shoot-ball',       # 9

        'shoot-bow',        # 10
        'walk',             # 11
        'unk',
    ],
}['ucfhmdb' if dataset in ['ucf', 'hmdb'] else dataset]


p_config = Path(r'work_dirs/train_output/ucf2hmdb/tsm/osbp/5652__osbp-tsm_ucf2hmdb/16/20220807-213529/osbp_tsm_ucf2hmdb.py')
# p_config = Path(r'work_dirs/train_output/hmdb2ucf/tsm/osbp/5649__osbp-tsm_hmdb2ucf/17/20220807-182029/osbp_tsm_hmdb2ucf.py')

p_preds = p_config.parent / 'best_pred.pkl'
with p_preds.open('rb') as f:
    preds = np.array(pickle.load(f))
    preds = preds.argmax(axis=1)

cfg = Config.fromfile(p_config)
with open(cfg['ann_file_test_target'], 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    data = np.array(list(reader))

labels = data[:,-1].astype(np.int64)
wrong = preds != labels
num_classes = labels.max() + 1

conf = confusion_matrix(preds, labels)
conf_wrong = conf.sum(axis=0) - conf.diagonal()
print(conf)
print('wrong:')
print('', conf_wrong)

data_prefix = cfg['data_prefix_target']
# data_prefix = '/data/dataset/' / Path('/'.join(data_prefix.split('/')[2:]))  # todo: for master
clip_infos = [[] for _ in range(num_classes)]
for pred, (path, length, label) in zip(preds[wrong], data[wrong]):
    if len(clip_infos[pred]) < 16:
        clip_infos[pred].append([Path(path), int(label)])

total_clips = sum(map(len, clip_infos))
count = 0
output_duration = 20  # secs
clips = [[] for _ in range(num_classes)]
for pred, row in enumerate(clip_infos):
    for p_video, label in row:
        p_images = sorted((data_prefix / p_video).glob('*.jpg'))
        assert all(p_iamge.is_file() for p_iamge in p_images)
        p_images = list(map(str, p_images))
        clip = ImageSequenceClip(p_images, fps=30.)
        txt_clip = TextClip(labelmap[label], fontsize=75, color='white')
        # https://github.com/Zulko/moviepy/issues/401#issuecomment-278679961
        txt_clip = txt_clip.set_pos('center').set_duration(clip.duration)
        clip = CompositeVideoClip([clip, txt_clip])
        if clip.duration < output_duration:
            clip = concatenate_videoclips([clip] * round(output_duration / clip.duration + .5))  # ceil
        clip = clip.set_duration(output_duration)
        clips[pred].append(clip)
        count += 1
        print(f'\r{count}/{total_clips}', end='')
print('\ndone')


height = 360
column = 4
p_outdir = Path(r'visualizations/failure_modes')
# if p_outdir.exists():
#     shutil.rmtree(p_outdir)
# p_outdir.mkdir()
for pred, clip in enumerate(clips):
    p_out = str(p_outdir / f'{pred:2d}_{labelmap[pred]}.mp4')
    if not clip:
        continue
    num_to_fill = (column - len(clip) % column) % column  # 0, 1, 2, 3
    for _ in range(num_to_fill):
        clip.append(ColorClip((height, height), (0,0,0), duration=output_duration))
    final_clip = clips_array(
        np.array(clip).reshape(-1, column)
    )
    final_clip = final_clip.resize(height=height)
    final_clip.write_videofile(p_out)
