# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import trunc_normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class TimeSformerHead(BaseHead):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, domains=None):
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

    def loss(self, cls_score, labels, domains=None, **kwargs):
        if domains is not None:
            source_idx = domains == 'source'
            target_idx = ~source_idx
            mca_source = self.calc_mca(cls_score[source_idx], labels[source_idx])
            mca_target = self.calc_mca(cls_score[target_idx], labels[target_idx])
            losses = {'mca_source': mca_source, 'mca_target': mca_target}
        else:
            mca = self.calc_mca(cls_score, labels)
            losses = {'mca': mca}

        loss_cls = self.loss_cls(cls_score, labels, domains=domains, **kwargs)
        losses['loss_cls'] = loss_cls
        return losses
