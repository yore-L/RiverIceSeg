import torch
from torch import nn, Tensor
import torch.nn.functional as F

_all__ = ["FocalCosineLoss","FocalLoss"]


class FocalCosineLoss(nn.Module):
    """
    Implementation Focal cosine loss from the "Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble" (https://arxiv.org/abs/2007.07805).

    Credit: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
    """

    def __init__(self, alpha: float = 1, gamma: float = 2, xent: float = 0.1, reduction="mean"):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        cosine_loss = F.cosine_embedding_loss(
            input,
            torch.nn.functional.one_hot(target, num_classes=input.size(-1)),
            torch.tensor([1], device=target.device),
            reduction=self.reduction,
        )

        cent_loss = F.cross_entropy(F.normalize(input), target, reduction="none")
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean", eps: float = 1e-6, ignore_index: int = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets
        if t.dim() == logits.dim() - 1:
            t = t.unsqueeze(1)
        t = t.type_as(logits)
        ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * t + (1 - p) * (1 - t)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = focal_term * ce
        if self.alpha is not None:
            loss = loss * (self.alpha * t + (1 - self.alpha) * (1 - t))
        if self.ignore_index is not None:
            mask = (t != self.ignore_index).float()
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / mask.sum().clamp_min(self.eps)
        return loss.mean() if self.reduction == "mean" else loss.sum()