from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import *

from pathlib import Path


def get_clip(
    p_rawframe_dir:Path,
    label_name:str,
    pred_name:str='',
    fontsize=75,
    duration:int=20
) -> ImageSequenceClip:

    p_images = sorted(p_rawframe_dir.glob('*.jpg'))
    assert all(p_iamge.is_file() for p_iamge in p_images)
    p_images = list(map(str, p_images))
    clip = ImageSequenceClip(p_images, fps=30.)

    if label_name is not None:
        text = f'{label_name}'
        text += f'\n({pred_name})' if pred_name else ''
        txt_clip = TextClip(text, fontsize=fontsize, color='white')
        # https://github.com/Zulko/moviepy/issues/401#issuecomment-278679961
        txt_clip = txt_clip.set_pos('center').set_duration(clip.duration)
        clip = CompositeVideoClip([clip, txt_clip])

    if clip.duration < duration:
        clip = concatenate_videoclips([clip] * round(duration / clip.duration + .5))  # ceil
    clip = clip.set_duration(duration)

    return clip
