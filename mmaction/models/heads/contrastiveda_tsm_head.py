import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init
import numpy as np

from ..builder import HEADS
from ...core import top_k_accuracy, confusion_matrix, mean_class_accuracy
from .base import AvgConsensus, BaseHead, get_fc_block


class RunningAverage:
    def __init__(self, length=30):
        self.mean = None
        self.stored = None
        self.length = length
    def update(self, X):
        with torch.no_grad():
            assert len(X.size()) == 2
            if self.stored is not None:
                self.stored = torch.cat([self.stored, X], dim=0)[-self.length:]
            else:
                self.stored = X[-self.length:]
            self.mean = self.stored.mean(dim=0)


@HEADS.register_module()
class ContrastiveDATSMHead(BaseHead):
    def __init__(self,
            num_classes,
            in_channels,
            num_layers=1,
            num_features=512,
            centroids=dict(p_centroid=''),  # p_centroid would be "PXX_train_closed.pkl"
            num_segments=8,
            debias=False,
            bias_input=False,
            bias_network=False,
            debias_last=True,
            hsic_factor=.5,
            linear_head=None,
            loss_cls=dict(type='SemisupervisedContrastiveLoss', unsupervised=True, loss_ratio=.35, tau=5.),
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.8,
            init_std=0.001,
            is_shift=True,
            temporal_pool=False,
            **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_segments = num_segments
        self.debias = debias
        self.bias_input = bias_input
        self.bias_network = bias_network
        self.debias_last = debias_last
        self.hsic_factor = hsic_factor

        self.linear_head_debug = linear_head is not None

        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        if self.debias:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if self.spatial_type == 'avg' else nn.Identity()
            self.f1_conv3d = ConvModule(
                in_channels,
                in_channels * 2, (1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d', requires_grad=True))
            self.fc_feature1 = get_fc_block(2 * self.in_channels, self.num_features, self.num_layers, self.dropout_ratio)
            if self.bias_input:
                self.f2_conv3d = ConvModule(
                    in_channels,
                    in_channels * 2, (1, 3, 3),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1),
                    bias=False,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d', requires_grad=True))
                self.fc_feature2 = get_fc_block(2 * self.in_channels, self.num_features, self.num_layers, self.dropout_ratio)
            if self.bias_network:
                self.f3_conv2d = ConvModule(
                    in_channels,
                    in_channels * 2, (3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN', requires_grad=True))
                self.fc_feature3 = get_fc_block(2 * self.in_channels, self.num_features, self.num_layers, self.dropout_ratio)
            self.hsic = HSICLoss()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1) if self.spatial_type == 'avg' else nn.Identity()
            self.fc_feature = get_fc_block(self.in_channels, self.num_features, self.num_layers, self.dropout_ratio)
            consensus_ = consensus.copy()
            self.consensus = AvgConsensus(**consensus_) if consensus_.pop('type') == 'AvgConsensus' else nn.Identity()

        if self.linear_head_debug:
            self.linear_head = nn.Linear(self.num_features, linear_head['num_classes'])
            self.ce = torch.nn.CrossEntropyLoss()

        if centroids['p_centroid']:
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
                normal_init(m, std=self.init_std)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def train(self, mode=True):
        if self.linear_head_debug:
            super().train(False)
            # self.freeze_stages()
        else:
            super().train(mode)
    
    def freeze_stages(self):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = False
        for param in self.linear_head.parameters():
            param.requires_grad = True

    def forward(self, f, num_segs, domains=None):
        # 4N: source view 1,2,1,2,1,...,2, target view 1,2,1,2,1,...,2
        if self.debias:
            f_2d = f  # [4N*segs, C, H, W]
            f_3d = f.view((-1, self.num_segments, *f.shape[-3:])).transpose(1, 2)  # [4N, C, segs, H, W]
            self.zs = []
            self.ffs = []

            # standard route
            ff1 = self.f1_conv3d(f_3d)  # [4N, 2C, segs, H', W']
            ff1 = self.avg_pool(ff1)  # [4N, 2C, 1, 1, 1]
            ff1 = torch.flatten(ff1, start_dim=1)  # [4N, 2C]
            z1 = self.fc_feature1(ff1)  # [4N, n_feat]
            self.zs.append(z1)
            self.ffs.append(ff1)

            if self.bias_input:
                ff2 = self.f2_conv3d(f_3d[:,:,torch.randperm(f.shape[2])])  # [4N, 2C, segs, H', W']
                ff2 = self.avg_pool(ff2)  # [4N, 2C, 1, 1, 1]
                ff2 = torch.flatten(ff2, start_dim=1)  # [4N, 2C]
                z2 = self.fc_feature2(ff2)  # [4N, n_feat]
                self.zs.append(z2)
                self.ffs.append(ff2)

            if self.bias_network:
                ff3 = self.f3_conv2d(f_2d)  # [4N*segs, 2C, H', W']
                ff3 = ff3.view((-1, self.num_segments, *ff3.shape[-3:])).transpose(1, 2)  # [4N, 2C, segs, H', W']
                ff3 = self.avg_pool(ff3)  # [4N, 2C, 1, 1, 1]
                ff3 = torch.flatten(ff3, start_dim=1)  # [4N, 2C]
                z3 = self.fc_feature3(ff3)  # [4N, n_feat]
                self.zs.append(z3)
                self.ffs.append(ff3)

            result = torch.stack(self.zs, dim=1)  # [4N, 1~3, n_feat]

        else:
            f = self.avg_pool(f)  # [4N*segs, C, 1, 1]
            f = torch.flatten(f, start_dim=1)  # [4N*segs, C]
            f = self.fc_feature(f)  # [4N*segs, n_feat]
            f = f.view((-1, self.num_segments, f.shape[-1]))  # [4N, segs, n_feat]
            f = self.consensus(f)  # [4N, 1, n_feat]
            result = f
        
        if self.linear_head_debug:
            # with torch.set_grad_enabled(True):  # turned off by recognizer2D
            #     result = self.linear_head(result.detach())
            result = self.linear_head(result)
        
        return result
        
    
    def loss(self, cls_score, labels, domains, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model. [4N, 1~3, n_feat]
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if self.linear_head_debug:
            return {'loss_linear_head_debug': self.ce(cls_score[:,0,:], labels)}

        losses = dict()
        N = cls_score.shape[0] // 4
        N_labeled = 2*N if self.loss_cls.unsupervised else -1

        # centroids are needed only for scoring
        if not self.with_given_centroids:
            for c in range(self.num_classes):
                self.centroids[c].update(cls_score[:N_labeled,0][labels[:N_labeled]==c])

        if not self.multi_class and cls_score.size() != labels.size():
            # log scores 
            with torch.no_grad():
                cls_score_labeled_view1 = cls_score[:N_labeled,0].unsqueeze(dim=1)  # [2N, 1, n_feat]
                if not self.with_given_centroids:
                    centroids = torch.stack([c.mean for c in self.centroids])
                else:
                    centroids = self.centroids
                centroids = centroids.unsqueeze(dim=0)  # [1, k, n_feat]
                distances = (cls_score_labeled_view1 - centroids) ** 2  # [2N, k, n_feat]
                distances = distances.mean(dim=2) ** .5  # [2N, k]
                mca = mean_class_accuracy(
                    (-distances).detach().cpu().numpy(),  # score := negative distance 
                    labels[:N_labeled].detach().cpu().numpy()
                )
                losses['mca'] = torch.tensor(mca, device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = [self.loss_cls(cls_score[:,i], labels, domains, **kwargs) for i in range(cls_score.shape[1])]  # list of dicts
        loss_cls = {k: sum([l[k] for l in loss_cls]) for k in ['loss_super', 'loss_unsuper'] if k in loss_cls[0]}  # sum each loss
        losses.update(loss_cls)
        if self.debias and (self.bias_input or self.bias_network):
            loss_hsic = self._hsic_loss(kwargs['iter'])
            losses['loss_hsic'] = loss_hsic
        # print(f'| {losses["loss_super"]:.3f}\t{losses["loss_unsuper"]:.3f}\t{losses["loss_hsic"]: .3f}')
        if losses["loss_super"].isnan() or losses.get("loss_unsuper", torch.tensor(0)).isnan() or losses.get("loss_hsic", torch.tensor(0)).isnan():
            # kill the parent process
            import os
            import signal
            print('\n\n\nkill this training process due to nan values\n\n\n')
            os.kill(os.getppid(), signal.SIGTERM)
        return losses

    def _hsic_loss(self, cur_iter: int):
        features = self.zs if self.debias_last else self.ffs
        assert len(features) > 1
        loss_hsic_f = self.hsic(features[0], features[1].detach())
        loss_hsic_h = -self.hsic(features[0].detach(), features[1])
        if len(features) == 3:
            loss_hsic_f += self.hsic(features[0], features[2].detach())
            loss_hsic_h += -self.hsic(features[0].detach(), features[2])
        mask = cur_iter % 2  # 0 or 1; training f and h alternatively
        loss = mask * loss_hsic_f + (1 - mask) * loss_hsic_h
        return self.hsic_factor * loss


class HSICLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)
        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def hsic_loss(self, input1, input2):
        N = len(input1)
        if N < 4:
            return torch.tensor(0.0).to(input1.device)
        # we simply use the squared dimension of feature as the sigma for RBF kernel
        sigma_x = np.sqrt(input1.size()[1])
        sigma_y = np.sqrt(input2.size()[1])

        # compute the kernels
        kernel_XX = self._kernel(input1, sigma_x)
        kernel_YY = self._kernel(input2, sigma_y)
        
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
            torch.trace(tK @ tL)
            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        return hsic

    def forward(self, x1, x2):
        return self.hsic_loss(x1, x2)
