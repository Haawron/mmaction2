from ..builder import LOSSES
from .base import BaseWeightedLoss

import torch
import torch.nn.functional as F


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
            target_domain_label=.5):
        super().__init__()
        self.num_classes = num_classes
        self.target_domain_label = target_domain_label

    def bce(self, pred, label):
        """
        Args:
            pred (N x 2): logits (positive, negative)
            label (N): soft labels

        Returns:
            Binary Cross Entropy (scalar)
        """
        q = 1. / (1.+torch.exp(pred[:,1]-pred[:,0]))  # N
        p = label  # N
        H = - p*torch.log(q) - (1-p)*torch.log(1-q)  # N
        H = H.mean()  # no dim
        return H
        
    def _forward(self, cls_score, label, **kwargs):
        """
        Args:
            cls_score (torch.Tensor, (N x clip_len) x (K + 1)): The K+1-dim class score (before softmax).
            label (torch.Tensor, N): The ground-truth label, hard-labeled(integers).
            kwargs:
                domain (required): 
                epoch: The number of epochs during training.
                total_epoch: The total epoch.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        assert 'domain' in kwargs
        domain = kwargs['domain']
        print('\n\n\n', domain, label, '\n\n\n')

        source_idx = torch.squeeze(domain == 0)
        target_idx = torch.logical_not(source_idx)

        soft_labels = torch.eye(self.num_classes).to(cls_score.device)
        soft_labels[-1] = self.target_domain_label * torch.ones(self.num_classes)
        label[target_idx] = -1
        y = soft_labels[label]

        label = label.unsqueeze(dim=1)
        if source_idx.any():
            lsm = F.log_softmax(cls_score[source_idx], dim=1)  # N x (K+1)
            loss_s = -(label[source_idx] * lsm).sum(dim=1)  # N
            loss_s = loss_s.mean()  # no dim
        else:
            loss_s = 0
        
        if target_idx.any():
            logit_known, logit_unknown = cls_score[target_idx,:-1], cls_score[target_idx,-1]
            logit_known = logit_known.sum(dim=1)
            loss_t = self.bce(
                torch.cat((logit_known.unsqueeze(1), logit_unknown.unsqueeze(1)), 1),
                label[target_idx])
        else:
            loss_t = 0

        return loss_s + loss_t
