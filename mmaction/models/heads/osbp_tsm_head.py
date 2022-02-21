import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

from ..builder import HEADS
from .tsm_head import TSMHead
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, target_idx_mask):
        ctx.save_for_backward(target_idx_mask)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        target_idx_mask, = ctx.saved_tensors
        grad_output[target_idx_mask] *= -1.
        return grad_output, None


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
    
    def forward(self, x, num_segs, domains):
        """
        Args:
            x (N x num_segs, c, h, w)
        
        Note:
            N: batch size
            num_segs: num_clips
        """
        target_idx_mask = torch.squeeze(domains == 1)
        target_idx_mask = target_idx_mask.repeat(num_segs)
        x = GradReverse.apply(x, target_idx_mask)
        return super().forward(x, num_segs)
