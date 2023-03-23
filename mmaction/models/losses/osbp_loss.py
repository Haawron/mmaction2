from ..builder import LOSSES
from .base import BaseWeightedLoss

import torch
import torch.nn.functional as F
import numpy as np


@LOSSES.register_module()
class OSBPLoss(BaseWeightedLoss):
    """OSBP Loss.

    Since OSBP sets the target domain label as [.5, .5, ..., .5], the label cannot be
    regarded as neither soft(not sum to 1) nor hard(single int) label. Thus, rather than re-using
    cross-entropy class mmaction provided, we define OSBPLoss.

    Args:
        num_classes: The total number of classes including the target domain class, which denoted as K+1.
        target_domain_label: The value of the target label. So the final
            label is `target_domain_label * torch.ones(num_classes+1)`
    """

    def __init__(
            self,
            num_classes,
            target_domain_label=.5,
            weight_loss_target=1e-3,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.target_domain_label = target_domain_label
        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()
        self.weight_loss_target = weight_loss_target
        
    def loss_adv(self, logits, t):
        p = F.softmax(logits, dim=1)[:,-1]  # being unknown
        t = t * torch.ones_like(p)  # expand
        return self.bce(p, t)
        
    def _forward(self, cls_score, label, domains=None, **kwargs):
        """
        Args:
            cls_score (torch.Tensor, (N x clip_len) x (K + 1)): The K+1-dim class score (before softmax).
            label (torch.Tensor, N): The ground-truth label, hard-labeled(integers).
            kwargs:
                epoch: The number of epochs during training.
                total_epoch: The total epoch.
        
        Returns:
            torch.Tensor: Computed loss.
        """

        source_idx = torch.from_numpy(domains == 'source')
        target_idx = torch.logical_not(source_idx)
        assert source_idx.sum() == target_idx.sum()
        logits_source = cls_score[source_idx]
        logits_target = cls_score[target_idx]
        labels_source = label[source_idx]
        loss_s = self.loss_cls(logits_source[:,:-1], labels_source)
        loss_t = self.loss_adv(logits_target, self.target_domain_label)

        # return loss_s + loss_t
        return {'loss_source_cls': loss_s, 'loss_target': self.weight_loss_target * loss_t}
