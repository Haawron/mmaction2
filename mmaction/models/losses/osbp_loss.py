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
            target_domain_label=.5,
            weight_loss=2.,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.target_domain_label = target_domain_label
        self.weighting_loss = weighting_loss
        self.loss = torch.nn.CrossEntropyLoss()
        self.weight_loss = weight_loss
        
    def _forward(self, cls_score, label, domains: torch.Tensor, **kwargs):
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
        # assert 'domain' in kwargs
        # domain = kwargs['domain']

        # source_idx = torch.squeeze(domain == 0, dim=1)
        # target_idx = torch.logical_not(source_idx)

        # if source_idx.any():
        #     loss_s = self.loss(cls_score[source_idx,:-1], label[source_idx])
        # else:
        #     loss_s = torch.tensor(0)
        
        # if target_idx.any():
        #     logit_known, logit_unknown = cls_score[target_idx,:-1], cls_score[target_idx,-1]
        #     logit_known = logit_known.sum(dim=1)
        #     predicted = torch.cat((logit_known.unsqueeze(1), logit_unknown.unsqueeze(1)), 1)
        #     soft_label = torch.tensor([[self.target_domain_label, self.target_domain_label]], device='cuda').repeat(target_idx.sum(), 1)
        #     loss_t = self.loss(predicted, soft_label)
        # else:
        #     loss_t = torch.tensor(0)

        # if self.weighting_loss:
        #     p = source_idx.sum() / (source_idx.sum() + target_idx.sum())
        # else:
        #     p = .5

        # return p * loss_s + (1-p) * loss_t

        source_idx = torch.from_numpy(domains == 'source')
        target_idx = torch.logical_not(source_idx)

        assert source_idx.sum() == target_idx.sum()

        loss_s = self.loss(cls_score[source_idx,:-1], label[source_idx])
        logit_known, logit_unknown = cls_score[target_idx,:-1], cls_score[target_idx,-1]
        logit_known = logit_known.sum(dim=1)
        predicted = torch.cat((logit_known.unsqueeze(1), logit_unknown.unsqueeze(1)), 1)
        soft_label = torch.tensor(
            [[self.target_domain_label, self.target_domain_label]], device='cuda').repeat(target_idx.sum(), 1)
        loss_t = self.loss(predicted, soft_label)

        # return loss_s + loss_t
        return {'loss_cls': loss_s, 'loss_target': self.weight_loss * loss_t}
