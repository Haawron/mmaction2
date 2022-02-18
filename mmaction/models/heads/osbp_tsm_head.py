import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

from ..builder import HEADS
from .tsm_head import TSMHead
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init


class GradReverse(Function):
    def __init__(self, lambd=1.):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


@HEADS.register_module()
class OSBPTSMHead(TSMHead):
    def __init__(self,
            num_classes,
            in_channels,
            num_segments=8,
            loss_cls=dict(type='CrossEntropyLoss'),
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.8,
            init_std=0.001,
            is_shift=True,
            temporal_pool=False,
            **kwargs):
        super().__init__(num_classes,
            in_channels,
            num_segments=num_segments,
            loss_cls=loss_cls,
            spatial_type=spatial_type,
            consensus=consensus,
            dropout_ratio=dropout_ratio,
            init_std=init_std,
            is_shift=is_shift,
            temporal_pool=temporal_pool,
            **kwargs)
        self.grl = GradReverse()
    
    def forward(self, x, *, labels):  # avoiding receiving num_segments legacy
        """
        Args:
            x ((N x num_clips) x c x h x w)
        """
        target_idx = labels == self.num_classes - 1
        x[target_idx] = self.grl(x[target_idx])
        super().forward(x)
