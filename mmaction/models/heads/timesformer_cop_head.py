from ..builder import HEADS
from .base import BaseHead

from mmcv.cnn import normal_init

import torch
import torch.nn as nn
from einops import rearrange

import math
from itertools import combinations


# @HEADS.register_module()
# class TimeSFormerCOPHead(BaseHead):
#     def __init__(self,
#         num_clips,
#         in_channels,
#         num_hidden=512,
#         loss_cls=dict(type='CrossEntropyLoss'),
#         dropout_ratio=0.8,
#         init_std=0.001,
#         **kwargs):
#         self.num_clips = num_clips
#         self.class_num = math.factorial(self.num_clips)
#         super().__init__(self.class_num, in_channels, loss_cls, **kwargs)
#         self.num_hidden = num_hidden

#         self.init_std = init_std

#         self.fc_cop1 = nn.Linear(self.in_channels*2, self.num_hidden)
#         self.pair_num = int(self.num_clips*(self.num_clips-1)/2)  # NC2
#         self.fc_cop2 = nn.Linear(self.num_hidden*self.pair_num, self.class_num)

#         self.dropout = nn.Dropout(p=dropout_ratio)
#         self.relu = nn.ReLU(inplace=True)

#     def init_weights(self):
#         for layer in [self.fc_cop1, self.fc_cop2]:
#             normal_init(layer, std=self.init_std)

#     def forward(self, f, domains=None):
#         # f: [2B*N, C_in]
#         f = f.reshape(-1, self.num_clips, self.in_channels)  # [2B, N, C_in]
#         f = f.transpose(1, 0)   # [N, 2B, C_in]
#         pf = [torch.cat([f[i], f[j]], dim=-1) for i, j in combinations(range(self.num_clips), 2)]  # [pair_num] * [2B, 2*C_in]
#         pf = torch.stack(pf, dim=1)  # [2B, pair_num, 2*C_in]
#         pf = pf.reshape(-1, 2*self.in_channels)  # [2B*pair_num, 2*C_in]

#         h = self.dropout(self.relu(self.fc_cop1(pf)))  # [2B*pair_num, num_hidden]
#         h = rearrange(h, '(bb p) h -> bb (p h)', p=self.pair_num)  # [2B, pair_num*num_hidden]
#         h = self.fc_cop2(h)  # logits: [2B, class_num]

#         return h

#     def loss(self, cls_score, labels, domains, **kwargs):
#         losses = super().loss(cls_score, labels)
#         return losses

# https://github.com/xudejing/video-clip-order-prediction/blob/master/models/vcopn.py
# @HEADS.register_module()
class VCOPN(BaseHead):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.fc_cop1 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc_cop2 = nn.Linear(512*pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]  # [2B, T, ]
            f.append(self.base_network(clip))

        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))

        pf = [self.fc_cop1(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc_cop2(h)  # logits

        return h
