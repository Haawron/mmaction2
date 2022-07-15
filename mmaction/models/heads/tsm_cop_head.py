from ..builder import HEADS
from .base import AvgConsensus, BaseHead
from ...core import confusion_matrix

from mmcv.cnn import normal_init

import numpy as np
import torch
import torch.nn as nn

import math
from itertools import combinations


@HEADS.register_module()
class TSMCOPHead(BaseHead):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self,
            num_clips,
            in_channels,
            num_hidden=512,
            num_segments=8,
            loss_cls=dict(type='CrossEntropyLoss'),
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.8,
            init_std=0.001,
            is_shift=True,
            **kwargs):
        """
        Args:
            num_features (int): 512
        """
        # todo: parameter 맞춰줘야 됨
        self.num_clips = num_clips
        self.class_num = math.factorial(self.num_clips)
        super().__init__(self.class_num, in_channels, loss_cls, **kwargs)
        self.num_segments = num_segments
        self.num_hidden = num_hidden

        self.init_std = init_std
        self.is_shift = is_shift
        self.spatial_type = spatial_type

        self.avg_pool = nn.AdaptiveAvgPool2d(1) if self.spatial_type == 'avg' else nn.Identity()

        self.fc7 = nn.Linear(self.in_channels*2, self.num_hidden)
        self.pair_num = int(self.num_clips*(self.num_clips-1)/2)  # NC2
        self.fc8 = nn.Linear(self.num_hidden*self.pair_num, self.class_num)

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in [self.fc7, self.fc8]:
            normal_init(layer, std=self.init_std)

    def forward(self, f, num_segs, domains=None):
        # f: [N*num_clips*segs, C_in, 7, 7]
        # 192 = [2(domains)*2(videos_per_gpu); permuted] * 3(num_clips; permuted) * 16(segs, clip_len)

        f = self.avg_pool(f)  # [N*num_clips*segs, C_in, 1, 1]
        f = torch.flatten(f, start_dim=1)  # [N*num_clips*segs, C_in]
        f = f.reshape(-1, self.num_clips, self.num_segments, self.in_channels)  # [N, num_clips, segs, C_in]
        f = f.transpose(1, 0)  # [num_clips, N, segs, C_in]

        # pair_num = comb(num_clips,2)
        # [pair_num] * [N, segs, 2*C_in]
        pf = [torch.cat([f[i], f[j]], dim=-1) for i, j in combinations(range(self.num_clips), 2)]  # dict order
        pf = torch.stack(pf, dim=1)  # [N, pair_num, segs, 2*C_in]
        pf = pf.reshape(-1, 2*self.in_channels)  # [N*pair_num*segs, 2*C_in]

        h = self.dropout(self.relu(self.fc7(pf)))  # [N*pair_num*segs, num_hidden]
        h = h.reshape(-1, self.pair_num, self.num_segments, self.num_hidden)  # [N, pair_num, segs, num_hidden]
        h = h.permute(0, 2, 1, 3)  # [N, segs, pair_num, num_hidden]
        h = h.reshape(-1, self.pair_num*self.num_hidden)  # [N*segs, pair_num*num_hidden]
        h = self.fc8(h)  # logits; [N*segs, class_num]
        h = h.reshape(-1, self.num_segments, self.class_num)  # [N, segs, class_num]
        h = h.mean(dim=1)  # average consensus; [N, class_num]

        return h
    
    def loss(self, cls_score, labels, domains, **kwargs):
        losses = super().loss(cls_score, labels)  # todo: dann은 train_ratio 있어도 잘 되는데 왜 여기서만 문제됨?
        return losses
