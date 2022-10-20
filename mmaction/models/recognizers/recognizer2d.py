# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1
            losses.update(loss_aux)

        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        if self.feature_extraction:
            # perform spatial pooling
            avg_pool = nn.AdaptiveAvgPool2d(1)
            x = avg_pool(x)
            # squeeze dimensions
            x = x.reshape((batches, num_segs, -1))
            # temporal average pooling
            x = x.mean(axis=1)
            return x

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`

        # should have cls_head if not extracting features
        cls_score = self.cls_head(x, num_segs)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

    def _do_fcn_test(self, imgs):
        # [N, num_crops * num_segs, C, H, W] ->
        # [N * num_crops * num_segs, C, H, W]
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = self.test_cfg.get('num_segs', self.backbone.num_segments)

        if self.test_cfg.get('flip', False):
            imgs = torch.flip(imgs, [-1])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
        else:
            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, fcn_test=True)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            assert not self.feature_extraction
            assert self.with_cls_head
            return self._do_fcn_test(imgs).cpu().numpy()
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        outs = self.cls_head(x, num_segs)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)


@RECOGNIZERS.register_module()
class DARecognizer2D(Recognizer2D):
    def __init__(self,
                 backbone=None,
                 cls_head=None,
                 neck=None,
                 contrastive=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(backbone,
                        cls_head=cls_head,
                        neck=neck,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg)
        self.contrastive = contrastive

    def _reshape_inputs(self, imgs, labels, domains):
        dim = imgs.dim()
        B = imgs.shape[0] // 2
        if self.contrastive:
            assert dim in [6, 7], f'Wrong input dimension {imgs.dim}. Should be 6 or 7'
            if dim == 6:  # NCHW (default)
                # [2B, 2, N, C, H, W] -> [2B * 2 * N, C, H, W]
                N, T = None, imgs.shape[2]  # N acts like T
            elif dim == 7:  # NCTHW (COP multi-task)
                # [2B, 2, N, C, T, H, W] -> [2B * 2 * N * T, C, H, W]
                N, T = imgs.shape[2], imgs.shape[4]
                imgs = imgs.transpose(3, 4)
            imgs = imgs.reshape((-1, ) + imgs.shape[-3:])
            labels = labels.reshape((-1, ) + labels.shape[2:])  # [2B, 2, 1] -> [2B*2, 1]
            domains = domains.reshape(-1)
        else:
            assert dim in [5, 6], f'Wrong input dimension {imgs.dim}. Should be 5 or 6'
            if dim == 5:  # NCHW
                # [2B, N, C, H, W] -> [2B * N, C, H, W]
                N, T = None, imgs.shape[1]
            elif dim == 6:  # NCTHW
                # [2B, N, C, T, H, W] -> [2B * N * T, C, H, W]
                N, T = imgs.shape[1], imgs.shape[3]
                imgs = imgs.transpose(2, 3)
            imgs = imgs.reshape((-1, ) + imgs.shape[-3:])
        self.B, self.N, self.T = B, N, T  # static values throughout the training
        gt_labels = labels.squeeze()  # [2B] or [2B*2]
        return imgs, gt_labels, domains

    def forward_train(self, imgs, labels, domains, **kwargs):
        assert self.with_cls_head
        imgs, gt_labels, domains = self._reshape_inputs(imgs, labels, domains)

        losses = dict()

        # X = 2B*2*T, 2B*2*N*T, 2B*T, 2B*N*T
        x = self.extract_feat(imgs)  # [X, C_feat, 7, 7]

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:  # auxiliary
            x, loss_aux = self.neck(x, None, domains, gt_labels)
            losses.update(loss_aux)

        cls_score = self.cls_head(x, None, domains)  # [2B(*2)(*N)(*T), 1~3, C, 7, 7]
        if self.N is not None:
            cls_score = cls_score.reshape(-1, self.N, *cls_score.shape[-2:])

        loss_cls = self.cls_head.loss(cls_score, gt_labels, domains, **kwargs)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, domains):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs, domains).cpu().numpy()

    def forward(self, imgs, label=None, domains=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)
            return self.forward_train(imgs, label, domains, **kwargs)

        return self.forward_test(imgs, domains, **kwargs)

    def train_step(self, data_batches, domains, optimizer=None, **kwargs):
        if self.cls_head.__dict__.get('linear_head_debug', False):  # todo: workaround for debugging, linear head debug recognizer를 따로 만들어야 될 듯
            imgs = data_batches['imgs']
            labels = data_batches['label']
            domains = torch.ones_like(imgs)
        else:
            imgs = torch.concat([data_batch['imgs'] for data_batch in data_batches])
            labels = torch.concat([data_batch['label'] for data_batch in data_batches])
            domains = np.array([
                [domain]*data_batch['imgs'].shape[0] for domain, data_batch in zip(domains, data_batches)
            ]).reshape(-1)

        if not self.contrastive:  # shuffle the batch
            indices = torch.randperm(imgs.shape[0])
            imgs = imgs[indices]
            labels = labels[indices]
            domains = domains[indices]

        losses = self(imgs, labels, domains, return_loss=True, **kwargs)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=imgs.shape[0])

        return outputs

    def _do_test(self, imgs, domains):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""

        # [B, N, C, T, H, W] or [B, T, C, H, W]
        if imgs.dim() == 6:
            B, N, C, T, H, W = imgs.shape
            imgs = imgs.transpose(2, 3)  # [B, N, T, C, H, W]
        elif imgs.dim() == 5:
            B, T, C, H, W = imgs.shape
        imgs = imgs.reshape(-1, *imgs.shape[-3:])

        # X = BNT or BT
        x = self.extract_feat(imgs)  # [X, C_feat, 7, 7]

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck and not self.neck.is_aux:  # auxiliary
            x, _ = self.neck(x, None, domains)

        if self.feature_extraction:
            x = nn.AdaptiveAvgPool2d(1)(x)  # [X, C_feat, 1, 1]
            x = torch.flatten(x, start_dim=1)  # [X, C_feat]
            x = x.reshape(-1, T, x.shape[-1])  # [*, T, C_feat]
            x = x.mean(axis=1)  # [*, C_feat]
            return x

        cls_score = self.cls_head(x, None, domains)
        # Shapes (N.B. ignore num_crops)
            # Vanilla:      [N, K]          (Not here, just for notice)
            # DANN: list of [N, K], [N, 1]
            # OSBP:         [N, K+1]
            # GCD:          [N, 1~3, n_feat]

        if type(cls_score) == list:  # {DANN}
            cls_score = cls_score[0]  # [N, K]
        elif len(cls_score.shape) == 3:  # {GCD4DA}
            cls_score = cls_score[:,0]  # [N, n_feat]

        return self.get_prob(cls_score)

    def get_prob(self, cls_score):
        # validate `average_clips`
        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')
        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', 'feature', None]:
            raise ValueError(f'{average_clips} is not supported. '
                            f'Currently supported ones are '
                            f'["score", "prob", "feature", None]')
        # Shapes
            # DANN: [N, K]
            # OSBP: [N, K+1]
            # GCD:  [N, n_feat]
        if average_clips in [None, 'feature']:
            return cls_score
        else:
            if self.contrastive:  # {GCD}
                cls_score = self.cls_head._get_logits_from_features(cls_score)

            if average_clips == 'score':
                return cls_score  # [N, K] or [N, K+1]
            elif average_clips == 'prob':
                return F.softmax(cls_score, dim=1)  # [N, K] or [N, K+1]
