import torch
from torch.autograd import Function
import numpy as np

from ..builder import HEADS
from ...core import top_k_accuracy
from .tsm_head import TSMHead


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
    
    def forward(self, x, num_segs, domains=np.array([])):
        """
        Args:
            x (N x num_segs, c, h, w)
        
        Note:
            N: batch size
            num_segs: num_clips
        """
        if domains.shape[0] > 0:
            target_idx_mask = torch.squeeze(torch.from_numpy(domains == 'target'))
            target_idx_mask = target_idx_mask.repeat(num_segs)
            x = GradReverse.apply(x, target_idx_mask)
        return super().forward(x, num_segs)

    def loss(self, cls_score, labels, domains, **kwargs):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score.device)

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