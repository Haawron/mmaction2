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
        self.loss = torch.nn.CrossEntropyLoss()
        
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

        source_idx = torch.squeeze(domain == 0)
        target_idx = torch.logical_not(source_idx)

        soft_labels = torch.eye(self.num_classes).to(cls_score.device)
        soft_labels[-1] = self.target_domain_label * torch.ones(self.num_classes)
        label[target_idx] = -1
        y = soft_labels[label]

        if source_idx.any():
            loss_s = self.loss(cls_score[source_idx,:-1], label[source_idx])
        else:
            loss_s = torch.tensor(0)
        
        if target_idx.any():
            logit_known, logit_unknown = cls_score[target_idx,:-1], cls_score[target_idx,-1]
            logit_known = logit_known.sum(dim=1)
            loss_t = self.loss(
                torch.cat((logit_known.unsqueeze(1), logit_unknown.unsqueeze(1)), 1),
                y[target_idx,:2])
        else:
            loss_t = torch.tensor(0)

        # print(f'{loss_s.item():.7f}\t{loss_t.item():.7f}')
        return loss_s + .01*loss_t
