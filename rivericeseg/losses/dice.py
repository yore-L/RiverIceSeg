import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional
__all__ = ["DiceLoss"]


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {"O", "M", "U", "S"}:
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

    raise ValueError("Unsupported input type" + str(type(x)))
class DiceLoss(nn.Module):
    """
    二分类专用的 Dice Loss 实现：
    计算分割结果与标签之间的 Dice coefficient 损失。
    输入:
      - logits: Tensor, shape [N, 1, H, W]
      - target: Tensor, shape [N, H, W] 或 [N, 1, H, W], 包含 {0,1} 和 ignore_index
    输出:
      - loss: 标量，Dice loss
    """
    def __init__(self, smooth: float = 0.0, ignore_index: Optional[int] = None, eps: float = 1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # 预测概率
        probs = torch.sigmoid(logits)
        # 对齐 target 形状
        if target.dim() == probs.dim() - 1:
            target = target.unsqueeze(1).float()
        else:
            target = target.float()
        # 掩码 ignore_index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            probs = probs * mask
            target = target * mask
        # 展平
        N = probs.size(0)
        probs = probs.view(N, -1)
        target = target.view(N, -1)
        # 计算交集与并集
        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        # Dice 系数
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth + self.eps)
        # 返回 Dice 损失
        return 1 - dice_score.mean()
