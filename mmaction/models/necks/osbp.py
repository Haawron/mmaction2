import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from einops import reduce

from ..builder import NECKS
from ..heads.osbp_tsm_head import GradReverse
from ..heads.base import get_fc_block_by_channels
from ..builder import build_loss


@NECKS.register_module()
class OSBP(nn.Module):
    def __init__(self,
        in_channels,
        num_classes,  # = K+1
        num_hidden_layers=1,
        dropout_ratio=0.5,
        init_std=0.001,
        target_domain_label=.5,
        loss_weight=1,
        as_head=False,
        backbone='TSM',  # TSM or TimeSformer
        num_segments=None,  # Only for TSM
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.as_head = as_head
        self.backbone = backbone
        assert self.backbone in ['TSM', 'TimeSformer']
        self.num_segments = num_segments

        self.fc_osbp:nn.Sequential = get_fc_block_by_channels(
            self.in_channels, self.num_classes, [self.in_channels] * self.num_hidden_layers,
            self.dropout_ratio)
        self.osbp_loss = build_loss(dict(
            type='OSBPLoss',
            num_classes=num_classes,
            target_domain_label=target_domain_label,
            loss_weight=loss_weight,
        ))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in self.fc_osbp:
            normal_init(layer, std=self.init_std)

    def forward(self, f, labels=None, domains=None, **kwargs):
        # f
            # TSM: [2B x N=8 x T=1, n_feat, H, W]
            # TimeSformer: [2B x N=1, n_feat=768]
        if self.backbone == 'TSM':
            ff = reduce(f, '(bb n t) c h w -> bb c', 'mean', n=self.num_segments, t=1).contiguous()
        else:
            ff = f

        if self.as_head and (domains is None or domains.shape[0] == 0):  # if this is a head and (valid or test)
            target_idx = torch.ones(ff.shape[0], device=ff.device)
        else:
            if ff.shape[0] == 2*domains.shape[0]:  # if contrastive
                domains = np.repeat(domains, 2)  # N <- 2N
            target_idx = torch.squeeze(torch.from_numpy(domains == 'target'))  # [N]
            ff = GradReverse.apply(ff, target_idx)

        cls_score = self.fc_osbp(ff)  # [2B, K+1]
        losses = self.osbp_loss(cls_score, labels, domains, **kwargs)

        if self.as_head:
            return cls_score, losses
        else:
            return f, losses
