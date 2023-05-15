_base_ = ['../../plain/__base__/gcd4da_phase0_tsm_ucfhmdb.py']

from configs.recognition.hello.gcd4da.plain.__base__.gcd4da_phase0_tsm_ucfhmdb import model

num_clips = 3  # N
clip_len = 16  # T

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

frame_sampler_train = dict(
    type='COPSampleFrames',
    num_clips=num_clips,
    clip_len=clip_len,
    clip_interval=8,
    frame_interval=1
)

frame_sampler_test = dict(
    type='COPSampleFrames',
    num_clips=1,
    clip_len=clip_len,
    clip_interval=1,
    frame_interval=1,
    test_mode=True
)

formatter = dict(type='FormatShape', input_format='NCTHW')


def processor(data:dict):
    data['train'][0]['pipeline'][0] = frame_sampler_train
    data['train'][1]['pipeline'][0] = frame_sampler_train
    data['val']['pipeline'][0] = frame_sampler_test
    data['test']['pipeline'][0] = frame_sampler_test
    data['videos_per_gpu'] = 1
    data['train'][0]['pipeline'][-3] = formatter
    data['train'][1]['pipeline'][-3] = formatter
    data['val']['pipeline'][-3] = formatter
    data['test']['pipeline'][-3] = formatter
    return data
