import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from einops import reduce

from ..builder import NECKS
from ..heads.dann_tsm_head import GradReverse
from ..heads.base import get_fc_block_by_channels


@NECKS.register_module()
class DomainClassifier(nn.Module):
    def __init__(self,
        in_channels,
        loss_weight=1.,
        num_layers=4,
        dropout_ratio=.5,
        init_std=0.001,
        backbone='TSM',  # TSM or TimeSformer
        num_segments=None,  # Only for TSM
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.loss_weight = loss_weight
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.backbone = backbone
        assert self.backbone in ['TSM', 'TimeSformer']
        self.num_segments = num_segments
        self.grl = lambda x: GradReverse.apply(x)
        self.fc_domain:nn.Sequential = get_fc_block_by_channels(
            self.in_channels, 1, [4096]*self.num_layers, self.dropout_ratio
        )

        self.bce = torch.nn.BCEWithLogitsLoss()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in self.fc_domain:
            normal_init(layer, std=self.init_std)

    def forward(self, f, labels=None, domains=None, **kwargs):
        # f:
            # TSM: [2B x N=8 x T=1, n_feat=2048, H=7, W=7]
            # TimeSformer: [2B, n_feat]
        if domains is None:  # if valid or test
            return f, None
        if self.backbone == 'TSM':
            ff = reduce(f, '(bb n t) c h w -> bb c', 'mean', n=self.num_segments, t=1)
        else:
            ff = f
        if ff.shape[0] == 2*domains.shape[0]:  # if contrastive
            domains = np.repeat(domains, 2)  # N <- 2N
        source_idx = torch.from_numpy(domains == 'source').to(f.device)  # [N]
        label_domain = source_idx.type(torch.float32)  # [N], 1 to source, 0 to target, which to which does not matter
        score_domain = self.fc_domain(self.grl(ff)).squeeze(dim=1)  # [N]
        loss_domain = self.loss_weight * self.bce(score_domain, label_domain)  # `score_domain` plays a role of logits
        domain_acc = ((score_domain > 0) == label_domain.type(torch.int32)).type(torch.float32).mean()
        return f, {'loss_dom': loss_domain, 'acc_dom': domain_acc}  # [N, 1]
