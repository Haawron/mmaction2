from .base import BaseHead
from .contrastiveda_transformer_head import ContrastiveDATransformerHead
from ..builder import HEADS

import torch.nn as nn
from torch.nn.init import trunc_normal_


@HEADS.register_module()
class DINOHead(ContrastiveDATransformerHead):
    def __init__(
        self, in_dim, out_dim,
        loss_cls=dict(type='CrossEntropyLoss'),
        use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **kwargs):

        super(ContrastiveDATransformerHead, self).__init__(12, in_dim, loss_cls, **kwargs)  # call grandparent method
        self._init_centroids()

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        # self.apply(self.init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def init_weights(self):
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.mlp:
            _init_weight(m)
        _init_weight(self.last_layer)

    def forward(self, x, num_segs=None, domains=None, gt_labels=None):
        # [2B*2, in_channels]
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
