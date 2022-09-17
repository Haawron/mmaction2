import torch
import torch.nn as nn
from mmcv.cnn import trunc_normal_init

from ..builder import HEADS
from ...core import mean_class_accuracy
from .base import BaseDAContrastiveHead, get_fc_block_by_channels
from .contrastiveda_tsm_head import RunningAverage


@HEADS.register_module()
class ContrastiveDATransformerHead(BaseDAContrastiveHead):
    def __init__(self,
                num_classes,
                in_channels,
                channels=[],
                num_features=512,
                loss_cls=dict(type='SemisupervisedContrastiveLoss', unsupervised=True, loss_ratio=.35, tau=5.),
                centroids=dict(p_centroid=''),  # p_centroid would be "PXX_train_closed.pkl"
                init_std=0.02,
                dropout_ratio=0.5,
                **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.num_features = num_features
        self.fc_contra = get_fc_block_by_channels(in_channels, num_features, channels, dropout_ratio)
        self._init_centroids(centroids)

    def _init_centroids(self, centroids={}):
        if centroids.get('p_centroid', None):
            self.with_given_centroids = True
            import pickle
            with open(centroids['p_centroid'], 'rb') as f:
                self.centroids = torch.from_numpy(pickle.load(f)).cuda()
            assert self.centroids.shape[0] == self.num_classes, f'Size mismatched: {(self.centroids.shape[0], self.num_classes)}'
        else:  # centroids are needed only for scoring
            self.with_given_centroids = False
            self.centroids = [RunningAverage() for _ in range(self.num_classes)]

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std)

    def forward(self, x, domains=None):
        # [4N, in_channels]
        cls_score = self.fc_contra(x)
        # [4N, num_classes]
        return cls_score

    # def loss(self, cls_score, labels, domains, **kwargs):
    #     """Calculate the loss given output ``cls_score``, target ``labels``.

    #     Args:
    #         cls_score (torch.Tensor): The output of the model. [4N, 1~3, n_feat]
    #         labels (torch.Tensor): The target output of the model.

    #     Returns:
    #         dict: A dict containing field 'loss_cls'(mandatory)
    #         and 'topk_acc'(optional).
    #     """

    #     if labels.shape == torch.Size([]):
    #         labels = labels.unsqueeze(0)
    #     elif labels.dim() == 1 and labels.size()[0] == self.num_classes and cls_score.size()[0] == 1:
    #         # Fix a bug when training with soft labels and batch size is 1.
    #         # When using soft labels, `labels` and `cls_socre` share the same
    #         # shape.
    #         labels = labels.unsqueeze(0)

    #     losses = dict()
    #     N = cls_score.shape[0] // 4
    #     N_labeled = 2*N if self.loss_cls.unsupervised else -1

    #     # centroids are needed only for scoring
    #     if not self.with_given_centroids:
    #         batch_labelset = set(labels[:N_labeled].detach().cpu().numpy())
    #         for c in batch_labelset:
    #             self.centroids[c].update(cls_score[:N_labeled][labels[:N_labeled]==c])

    #     if not self.multi_class and cls_score.size() != labels.size():
    #         # log scores
    #         with torch.no_grad():
    #             cls_score_labeled_view1 = cls_score[:N_labeled].unsqueeze(dim=1)  # [2N, 1, n_feat]
    #             if not self.with_given_centroids:
    #                 centroids = torch.stack([
    #                     # c.mean is None: never updated
    #                     c.mean if c.mean is not None else torch.zeros_like(cls_score[0])
    #                     for c in self.centroids
    #                 ])
    #             else:
    #                 centroids = self.centroids
    #             centroids = centroids.unsqueeze(dim=0)  # [1, k, n_feat]
    #             distances = (cls_score_labeled_view1 - centroids) ** 2  # [2N, k, n_feat]
    #             distances = distances.mean(dim=2) ** .5  # [2N, k]
    #             mca = mean_class_accuracy(
    #                 (-distances).detach().cpu().numpy(),  # score := negative distance
    #                 labels[:N_labeled].detach().cpu().numpy()
    #             )
    #             losses['mca_source'] = torch.tensor(mca, device=cls_score.device)

    #     elif self.multi_class and self.label_smooth_eps != 0:
    #         labels = ((1 - self.label_smooth_eps) * labels +
    #                   self.label_smooth_eps / self.num_classes)

    #     loss_cls = self.loss_cls(cls_score, labels, domains, **kwargs)  # dict

    #     # loss_cls = {k: sum([l[k] for l in loss_cls]) for k in ['loss_super', 'loss_unsuper'] if k in loss_cls}  # sum each loss
    #     losses.update(loss_cls)
    #     if losses["loss_super"].isnan() or losses.get("loss_unsuper", torch.tensor(0)).isnan():
    #         # kill the parent process
    #         import os
    #         import signal
    #         print('\n\n\nkill this training process due to nan values\n\n\n')
    #         os.kill(os.getppid(), signal.SIGTERM)
    #     return losses
