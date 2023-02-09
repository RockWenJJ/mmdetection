# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.fft as fft

from ..builder import LOSSES
from .utils import weighted_loss

def fft_loss(pred, target, reduction=True):
    pred_fft = fft.fftn(pred, dim=(2, 3))
    targ_fft = fft.fftn(target, dim=(2, 3))
    if reduction:
        loss = torch.mean(torch.log(torch.abs(pred_fft - targ_fft)+1.0))
    else:
        loss = torch.sum(torch.log(torch.abs(pred_fft - targ_fft)+1.0))
    return loss

@LOSSES.register_module()
class FFT2dLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(FFT2dLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        
    def forward(self,
                pred,
                target,
                reduction=True):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = self.loss_weight * fft_loss(
            pred, target, reduction=reduction)
        return loss