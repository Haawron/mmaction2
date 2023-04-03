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
            loss_weight=1,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.target_domain_label = target_domain_label
        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()
        self.loss_weight = loss_weight

    def loss_adv(self, p, t:float):  # p: [B]
        t = t * torch.ones_like(p)  # [B]
        return self.bce(p, t)

    def _forward(self, cls_score, labels, domains=None, **kwargs):
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
        if domains is None:  # valid or test
            return {}
        
        source_idx = torch.from_numpy(domains == 'source')
        target_idx = torch.logical_not(source_idx)
        assert source_idx.sum() == target_idx.sum()
        logits_source = cls_score[source_idx]
        logits_target = cls_score[target_idx]
        labels_source = labels[source_idx]
        prob_target = F.softmax(logits_target, dim=1)  # [B, K+1]
        loss_s = self.loss_cls(logits_source[:,:-1], labels_source)
        loss_t = self.loss_adv(prob_target[:,-1], self.target_domain_label)
        losses = {'loss_cls': loss_s, 'loss_osbp': self.loss_weight * loss_t}

        # for logging
        # unknown accuracy
        labels_target = labels[target_idx]  # [B]
        preds_target = prob_target.argmax(dim=1)  # [B]
        is_unknown_labels_target = labels_target >= self.num_classes - 1  # is unknown
        is_unknown_preds_target = preds_target >= self.num_classes - 1
        acc_unk = (is_unknown_preds_target == is_unknown_labels_target).type(torch.float32).mean()
        losses.update({'acc_unk': acc_unk})

        # mca
        mca_source = self.calc_mca(logits_source, labels_source)
        mca_target = self.calc_mca(logits_target, labels_target)
        losses.update({'mca_source': mca_source, 'mca_target': mca_target})
        # if domains is not None:  # train
        #     source_idx = torch.from_numpy(domains == 'source')
        #     target_idx = torch.logical_not(source_idx)
        #     assert source_idx.sum() == target_idx.sum()
        #     logits_source = cls_score[source_idx]
        #     logits_target = cls_score[target_idx]
        #     labels_source = labels[source_idx]
        #     labels_target = labels[target_idx]
        #     mca_source = self.calc_mca(logits_source, labels_source)
        #     mca_target = self.calc_mca(logits_target, labels_target)
        #     losses = {'mca_source': mca_source, 'mca_target': mca_target}
            
        #     prob_target = F.softmax(logits_target, dim=1)  # [B, K+1]
        #     loss_s = self.loss_cls(logits_source[:,:-1], labels_source)
        #     loss_t = self.loss_adv(prob_target[:,-1], self.target_domain_label)
        # else:  # valid or test
        #     return {}
        #     mca = self.calc_mca(cls_score, labels)
        #     losses = {'mca': mca}
        # losses.update({'acc_unk': acc_unk})
        return losses
