from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SoftCrossEntropyLoss"]

def _binary_label_smoothing(target: Tensor, eps: float, ignore_index: Optional[int]) -> Tensor:
    """
    对二分类标签做标签平滑：
      y' = y * (1 - eps) + 0.5 * eps
    并对 ignore_index 的位置 mask 掉。
    """
    # target: [N, H, W] 或 [N,1,H,W]
    # 统一至 [N,1,...]
    if target.dim() == target.unsqueeze(1).dim() - 1:
        tgt = target.unsqueeze(1).float()
    else:
        tgt = target.float().unsqueeze(1)
    smoothed = tgt * (1.0 - eps) + 0.5 * eps
    if ignore_index is not None:
        mask = (target != ignore_index).float().unsqueeze(1)
        smoothed = smoothed * mask
    return smoothed

class SoftCrossEntropyLoss(nn.Module):
    """
    支持标签平滑的 CrossEntropyLoss：
    - 多分类（C > 2）时调用原 label_smoothed_nll_loss
    - 二分类（C == 1 或 C == 2）时用 BCEWithLogits + 平滑标签
    """
    __constants__ = ["reduction", "ignore_index", "smooth_factor", "dim"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: float = 0.0,
        ignore_index: Optional[int] = -100,
        dim: int = 1,
    ):
        super().__init__()
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Args:
          input: logits, 形状 [N, C, ...]
          target: 整数标签，形状 [N, ...] 或 [N,1,...]
        """
        C = input.size(self.dim)
        # 二分类判断: 输出通道 1 或 2
        is_binary = (C == 1) or (C == 2)

        if is_binary:
            # 对于 C==2，可选取前景-背景差或只用前景通道
            if C == 2:
                # 差值作为单通道 logits
                logits = input[:, 1:2, ...] - input[:, 0:1, ...]
            else:
                logits = input

            # 去掉多余 singleton 维
            while logits.dim() > target.dim() + 1:
                logits = logits.squeeze(self.dim + 1)

            # 标签平滑
            smoothed = _binary_label_smoothing(target, self.smooth_factor, self.ignore_index)
            loss = F.binary_cross_entropy_with_logits(
                logits, smoothed, reduction="none"
            )

            # mask ignore_index
            if self.ignore_index is not None:
                mask = (target != self.ignore_index).float().unsqueeze(1)
                loss = loss * mask

        else:
            # 多分类路径
            log_prob = F.log_softmax(input, dim=self.dim)
            from .functional import label_smoothed_nll_loss
            loss = label_smoothed_nll_loss(
                log_prob,
                target,
                epsilon=self.smooth_factor,
                ignore_index=self.ignore_index,
                reduction="none",
                dim=self.dim,
            )

        # reduction
        if self.reduction == "mean":
            total = loss.numel()
            return loss.sum() / max(total, 1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
