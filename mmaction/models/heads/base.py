# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ...core import top_k_accuracy, mean_class_accuracy
from ..builder import build_loss

        
def get_fc_block(c_in, c_out, num_layers, dropout_ratio):
    fc_block = []
    for i in range(num_layers):
        if i < num_layers - 1:
            c_out_ = c_in // 2**(i+1)
            fc = nn.Linear(c_in//2**i, c_out_)
        else:
            c_out_= c_out
            fc = nn.Linear(c_in//2**(num_layers-1), c_out_)
        dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        act = nn.ReLU() if i < num_layers - 1 else nn.Identity()
        fc_block += [fc, dropout, act]
    return nn.Sequential(*fc_block)


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
        topk (int | tuple): Top-k accuracy. Default: (1, 5).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 topk=(1, 5), print_mca=False):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk
        self.print_mca = print_mca

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size() and self.topk is not None:
            top_k_acc = self.calc_topk(cls_score, labels, self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = a

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)
        
        if self.print_mca:
            mca = self.calc_mca(cls_score, labels)
            losses['mca'] = mca

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses

    def calc_topk(self, cls_score, labels, topk):
        top_k_acc = top_k_accuracy(
            cls_score.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
            topk)
        return [torch.tensor(acc, device=cls_score.device) for acc in top_k_acc]

    def calc_mca(self, cls_score, labels):
        mca = mean_class_accuracy(
            cls_score.detach().cpu().numpy(),
            labels.detach().cpu().numpy()
        )
        return torch.tensor(mca, device=cls_score.device)

    def unflatten_based_on_shiftedness(self, x):
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            return x.view((-1, self.num_segments // 2) + x.size()[1:])
        else:
            # [N, num_segs, num_classes]
            return x.view((-1, self.num_segments) + x.size()[1:])
