import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from torch.autograd import Function
import numpy as np

from ..builder import HEADS
from ...core import top_k_accuracy
from .base import AvgConsensus, BaseHead

from itertools import chain


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output *= -1.
        return grad_output


@HEADS.register_module()
class DANNTSMHead(BaseHead):
    def __init__(self,
            num_classes,
            in_channels,
            num_layers=1,
            num_segments=8,
            loss_cls=dict(type='CrossEntropyLoss'),
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.8,
            init_std=0.001,
            is_shift=True,
            temporal_pool=False,
            **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_layers = num_layers
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()
        consensus_type = consensus_.pop('type')

        if consensus_type == 'AvgConsensus':
            self.consensus_cls = AvgConsensus(**consensus_)
            self.consensus_domain = AvgConsensus(**consensus_)
        else:
            self.consensus_cls = nn.Identity()
            self.consensus_domain = nn.Identity()
        
        def get_fc_block(c_in, c_out):
            fc_block = []
            for i in range(self.num_layers):
                if i < self.num_layers - 1:
                    fc = nn.Linear(c_in//2**i, c_in//2**(i+1))
                else:
                    fc = nn.Linear(c_in//2**(self.num_layers-1), c_out)
                dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity()
                act = nn.ReLU() if i < self.num_layers - 1 else nn.Identity()
                fc_block += [fc, dropout, act]
            return nn.Sequential(*fc_block)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if self.spatial_type == 'avg' else nn.Identity()
        self.grl = lambda x: GradReverse.apply(x)
        self.fc_cls = get_fc_block(self.in_channels, self.num_classes)
        self.fc_domain = get_fc_block(self.in_channels, 1)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in chain(self.fc_cls, self.fc_domain):
            normal_init(layer, std=self.init_std)
    
    def forward(self, f, num_segs, domains=None):
        """
        Args:
            f (N x num_segs, c, h, w)
        
        Note:
            N: batch size
            num_segs: num_clips
        """

        # [N * num_segs, in_channels, 7, 7]
        f = self.avg_pool(f)
        # [N * num_segs, in_channels, 1, 1]
        f = torch.flatten(f, start_dim=1)  # squeeze??
        # [N * num_segs, in_channels]
        self.source_idx = (domains == 'source') if domains is not None else np.ones(f.shape[0]//self.num_segments, dtype=bool)
        repeated_source_idx = np.repeat(self.source_idx, self.num_segments)  # same as torch.repeat_interleave
        cls_score = self.fc_cls(f[repeated_source_idx])
        domain_score = self.fc_domain(self.grl(f))
        # [N * num_segs, num_classes]
        cls_score = self.unflatten_based_on_shiftedness(cls_score)
        domain_score = self.unflatten_based_on_shiftedness(domain_score)
        # [2 * N, num_segs // 2, num_classes] or [N, num_segs, num_classes]
        cls_score = self.consensus_cls(cls_score)
        domain_score = self.consensus_domain(domain_score)
        # [N, 1, num_classes]
        return [cls_score.squeeze(dim=1), domain_score.squeeze(dim=1)]  # [N, num_classes]x2

    def loss(self, cls_score, labels, domains, **kwargs):
        losses = dict()
        labels = labels[self.source_idx]
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score[0].size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score[0].detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score[0].device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, domains, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses

    def unflatten_based_on_shiftedness(self, x):
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            return x.view((-1, self.num_segments // 2) + x.size()[1:])
        else:
            # [N, num_segs, num_classes]
            return x.view((-1, self.num_segments) + x.size()[1:])
