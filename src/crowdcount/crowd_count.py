"""Crowd Counting Network"""
import torch
from torch import nn

from crowdcount import network
from crowdcount.models import CMTL


class CrowdCounter(nn.Module):
    """Crowd Counting Network"""

    def __init__(self, ce_weights=None):
        """Initialzie Crowd Counting Network"""
        super().__init__()
        self.CCN = CMTL()
        if ce_weights is not None:
            ce_weights = torch.Tensor(ce_weights)
            ce_weights = ce_weights.cuda()
        self.loss_mse_fn = nn.MSELoss()
        self.loss_bce_fn = nn.BCELoss(weight=ce_weights)

    def forward(self, im_data, gt_data=None, gt_cls_label=None, ce_weights=None):
        """Forward pass of Crowd Counting Network"""
        im_data = network.np_to_variable(
            im_data, is_cuda=False, is_training=self.training
        )
        density_map, _ = self.CCN(im_data)

        return density_map

    def build_loss(
        self,
        density_map,
        density_cls_score,
        gt_data,
        gt_cls_label,
        ce_weights,
    ):
        """Build loss function"""
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        ce_weights = torch.Tensor(ce_weights)
        ce_weights = ce_weights.cuda()
        cross_entropy = self.loss_bce_fn(density_cls_score, gt_cls_label)
        return loss_mse, cross_entropy
