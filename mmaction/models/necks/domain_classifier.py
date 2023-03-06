import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import NECKS
from ..heads.dann_tsm_head import GradReverse
from ..heads.base import get_fc_block


@NECKS.register_module()
class DomainClassifier(nn.Module):
    def __init__(self,
        in_channels,
        loss_weight=.5,
        num_layers=4,
        dropout_ratio=0.8,
        init_std=0.001,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.loss_weight = loss_weight
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.grl = lambda x: GradReverse.apply(x)
        self.fc_domain:nn.Sequential = get_fc_block(self.in_channels, 1, self.num_layers, self.dropout_ratio)

        self.bce = torch.nn.BCEWithLogitsLoss()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in self.fc_domain:
            normal_init(layer, std=self.init_std)

    def forward(self, f, labels=None, domains=None, **kwargs):
        if domains is None:  # if valid or test
            return f, None
        if f.shape[0] == 2*domains.shape[0]:  # if contrastive
            domains = np.repeat(domains, 2)  # N <- 2N
        source_idx = torch.from_numpy(domains == 'source')  # [N]
        label_domain = source_idx.type(torch.float32).to(f.device)  # [N], 1 to source, 0 to target, which to which does not matter
        score_domain = self.fc_domain(self.grl(f)).squeeze(dim=1)  # [N]
        loss_domain = self.loss_weight * self.bce(score_domain, label_domain)  # `score_domain` plays a role of logits
        return f, {'loss_domain': loss_domain}  # [N, 1]
