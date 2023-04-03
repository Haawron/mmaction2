from pathlib import Path

import cv2
import numpy as np
from einops import rearrange

from ..builder import PIPELINES


@PIPELINES.register_module()
class DebugInterPipelineVisualizer:
    def __init__(self, stop=False, namespace='clip_viz'):
        """An example of use case

        pipelines = [
            ...,
            dict(type='DebugInterPipelineVisualizer', namespace='clip_viz/01_before', stop=False),
            ...
            dict(type='DebugInterPipelineVisualizer', namespace='clip_viz/02_after', stop=True),
            ...
        ]
        """
        self.stop = stop
        self.namespace = namespace

    def __call__(self, results):
        assert 'imgs' in results
        imgs = results['imgs']  # (N x T) x [H, W, C], no batch size: each worker processes a single data
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        assert len(imgs) == num_clips * clip_len, f'{len(imgs)} != {num_clips} x {clip_len}'
        print(f'\n\nimage shape: ({num_clips} x {clip_len} = {len(imgs)}) x {imgs[0].shape}')

        p_tmp = Path(f'tmp/{self.namespace}'); p_tmp.mkdir(parents=True, exist_ok=True)
        imgs = np.array(imgs)
        imgs = rearrange(imgs, '(n t) h w c -> (n h) (t w) c', n=num_clips, t=clip_len)
        p_clip = p_tmp / f'hi.png'
        print(results['frame_inds'].reshape(num_clips, clip_len))
        print(f'\n\nStored the result in {p_clip}')
        cv2.imwrite(str(p_clip), cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))

        if self.stop:
            print('\n\nmmaction/datasets/pipelines/visualizer.py:DebugInterPipelineVisualizer stops the process\n')
            exit()
        else:
            print('\n\n')
            return results
