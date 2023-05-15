_base_ = ['../plain/gcd4da_phase0_tsm_hmdb2ucf.py']

from configs.recognition.hello.gcd4da.plain.__base__.gcd4da_phase0_tsm_ucfhmdb import model
from configs.recognition.hello.gcd4da.plain.gcd4da_phase0_tsm_hmdb2ucf import data

num_clips = 3  # N
clip_len = 8  # T

model['neck'] = dict(
    type='TSMCOPHead',

    as_neck=True,
    is_aux=True,

    loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.),
    num_clips=num_clips,
    num_segments=clip_len,
    in_channels=2048,

    dropout_ratio=0.1,
    init_std=0.001,
    is_shift=True,
)


data['train'][0]['pipeline'][0] = dict(
    type='COPSampleFrames',
    num_clips=num_clips,
    clip_len=clip_len,
    clip_interval=8,
    frame_interval=1
)
data['train'][1]['pipeline'][0] = dict(
    type='COPSampleFrames',
    num_clips=num_clips,
    clip_len=clip_len,
    clip_interval=8,
    frame_interval=1
)
data['val']['pipeline'][0] = dict(
    type='COPSampleFrames',
    num_clips=1,
    clip_len=clip_len,
    clip_interval=1,
    frame_interval=1,
    test_mode=True
)
data['test']['pipeline'][0] = dict(
    type='COPSampleFrames',
    num_clips=1,
    clip_len=clip_len,
    clip_interval=1,
    frame_interval=1,
    test_mode=True
)

data['videos_per_gpu'] = 2
data['train'][0]['pipeline'][-3] = dict(type='FormatShape', input_format='NCTHW')
data['train'][1]['pipeline'][-3] = dict(type='FormatShape', input_format='NCTHW')
data['val']['pipeline'][-3] = dict(type='FormatShape', input_format='NCTHW')
data['test']['pipeline'][-3] = dict(type='FormatShape', input_format='NCTHW')
