import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss

@LOSSES.register_module()
class CosineLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.
        
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert pred.shape == target.shape, 'prediction and target should have the same shape'
        if len(pred.shape) == 4: # B, C, H, W
            pred = pred.flatten(2).transpose(1, 2)
            target = target.flatten(2).transpose(1, 2)
        assert len(pred.shape) == 3
        pred_norm = F.normalize(pred, dim=-1)[..., None, :]
        targ_norm = F.normalize(target, dim=-1)[..., None, :]
        loss_cos = self.loss_weight * torch.mean(1. - pred_norm @ targ_norm.transpose(-2, -1))
        return loss_cos
        
        