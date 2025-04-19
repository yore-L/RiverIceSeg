from .dice import *
from .focal_cosine import *
from .edge import *

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

__all__ = ["JointLoss", "WeightedLoss", "CombinedBinaryLoss"]
class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


class CombinedBinaryLoss(nn.Module):
    def __init__(self,
                 focal_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 aux_weight: float = 0.3,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = "mean",
                 ignore_index: int = None):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.dice_fn = DiceLossV2(ignore_index=ignore_index)
        self.boundary_fn = EdgeLoss(ignore_index=ignore_index)
        self.focal_fn = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_logits: torch.Tensor = None) -> torch.Tensor:
        t = targets
        if t.dim() == logits.dim() - 1:
            t = t.unsqueeze(1)
        fl = self.focal_fn(logits, t)
        dl = self.dice_fn(logits, t)
        bd = self.boundary_fn(logits, t)
        main = self.focal_weight * fl + self.dice_weight * dl + bd
        if aux_logits is not None:
            afl = self.focal_fn(aux_logits, t)
            adl = self.dice_fn(aux_logits, t)
            abd = self.boundary_fn(aux_logits, t)
            aux = self.focal_weight * afl + self.dice_weight * adl + abd
            main = main + self.aux_weight * aux
        return main