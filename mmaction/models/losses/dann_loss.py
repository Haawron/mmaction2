from ..builder import LOSSES
from .base import BaseWeightedLoss

import torch
import torch.nn.functional as F
import numpy as np


@LOSSES.register_module()
class DANNLoss(BaseWeightedLoss):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.domain_loss = torch.nn.BCEWithLogitsLoss()
        
    def _forward(self, cls_score, label, domains=None, **kwargs):
        """
        Args:
            cls_score (list[torch.Tensor], 2 x ((N x clip_len) x (K + 1))): The K+1-dim class score (before softmax).
            label (torch.Tensor, N): The ground-truth label, hard-labeled(integers).
            kwargs:
                epoch: The number of epochs during training.
                total_epoch: The total epoch.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        source_idx = torch.from_numpy(domains == 'source')  # where domains be assigned: mmaction/core/runner/domain_adaptation_runner.py
        target_idx = torch.logical_not(source_idx)

        assert source_idx.sum() == target_idx.sum()
        
        cls_scores = cls_score  # can't edit the argument's name, so edited here
        cls_score, domain_score = cls_scores
        loss_cls = self.cls_loss(cls_score, label)
        loss_domain = self.domain_loss(domain_score, source_idx.type(torch.float32).to('cuda').unsqueeze(1))

        return loss_cls + loss_domain
