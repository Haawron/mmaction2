import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import NECKS
from ..heads.osbp_tsm_head import GradReverse
from ..heads.base import get_fc_block_by_channels
from ..builder import build_loss


@NECKS.register_module()
class OSBP(nn.Module):
    def __init__(self,
        in_channels,
        num_classes,
        num_hidden_layers=1,
        dropout_ratio=0.5,
        init_std=0.001,
        target_domain_label=.5,
        weight_loss_target=1e-3,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.fc_osbp:nn.Sequential = get_fc_block_by_channels(
            self.in_channels, self.num_classes, [self.in_channels] * self.num_hidden_layers, 
            self.dropout_ratio)
        self.osbp_loss = build_loss(dict(
            type='OSBPLoss',
            num_classes=num_classes,
            target_domain_label=target_domain_label,
            weight_loss_target=weight_loss_target,
        ))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in self.fc_osbp:
            normal_init(layer, std=self.init_std)

    def forward(self, f, labels=None, domains=None, **kwargs):
        if domains is None or domains.shape[0] == 0:  # if valid or test
            return f, None
        if f.shape[0] == 2*domains.shape[0]:  # if contrastive
            domains = np.repeat(domains, 2)  # N <- 2N

        target_idx = torch.squeeze(torch.from_numpy(domains == 'target'))  # [N]
        x = GradReverse.apply(f, target_idx)
        cls_score = self.fc_osbp(x)
        osbp_loss:dict = self.osbp_loss(cls_score, labels, domains)
        return f, osbp_loss
