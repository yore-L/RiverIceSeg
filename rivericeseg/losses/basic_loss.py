import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import cv2

# -------------------------------
# Edge Loss
# -------------------------------
class BoundaryLoss(nn.Module):
    """
    二分类边界损失，通过形态学梯度检测边缘
    """
    def __init__(self, egde_weight=3.0, kernel_size=3, ignore_index=None):
        super().__init__()
        self.egde_weight = egde_weight
        self.kernel_size = torch.ones(1, 1, kernel_size, kernel_size)
        self.ignore_index = ignore_index

        def get_edge_mask(self, targets: Tensor) -> Tensor:
            """
            生成边缘权重掩码
            输入形状：[N,1,H,W]
            输出形状：[N,1,H,W]
            """
            edge_masks = []
            for batch_idx in range(targets.shape):
                mask = targets[batch_idx, 0].cpu().numpy()

                # 形态学梯度检测边缘
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated = cv2.dilate(mask, kernel, iterations=1)
                eroded = cv2.erode(mask, kernel, iterations=1)
                edge = (dilated - eroded).astype(bool)

                # 转换为tensor并加权
                edge_tensor = torch.from_numpy(edge).to(targets.device)
                edge_masks.append(edge_tensor)

            edge_mask = torch.stack(edge_masks).unsqueeze(1).float()
            return edge_mask * self.edge_weight + 1.0  # 边缘区域权重更高

        def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
            # 生成边缘权重掩码
            edge_mask = self.get_edge_mask(targets)

            # 计算加权BCE损失
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                weight=edge_mask,
                reduction='none'
            )

            # 处理ignore_index
            if self.ignore_index is not None:
                valid_mask = (targets != self.ignore_index).float()
                bce_loss = bce_loss * valid_mask

            return bce_loss.mean()

# -------------------------------
# Focal Loss for Binary Classification
# -------------------------------
def focal_loss_with_logits(
        output: Tensor,
        target: Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        eps: float = 1e-6,
        ignore_index: int = None,
) -> Tensor:
    """
    计算二分类任务的 Focal Loss。

    参数：
        output: 模型的 logits，形状为 [N, 1, H, W]
        target: 二值标签，形状为 [N, H, W] 或 [N, 1, H, W]，取值 0、1 或 ignore_index（例如255）
        gamma: Focal loss 的幂指数
        alpha: 平衡正负样本的因子
        reduction: 'mean' 或 'sum'
        eps: 数值稳定性参数
        ignore_index: 忽略计算损失的标签值（例如255），若为 None，则不忽略

    返回：
        loss: 标量损失
    """
    # 如果 target 缺少通道维度，则 unsqueeze 到 [N, 1, H, W]
    if target.dim() == output.dim() - 1:
        target = target.unsqueeze(1)

    target = target.type_as(output)

    # 如果设置了 ignore_index，则构造 mask
    if ignore_index is not None:
        valid_mask = (target != ignore_index).float()
    else:
        valid_mask = torch.ones_like(target)

    p = torch.sigmoid(output)
    ce_loss = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    focal_term = (1.0 - pt).pow(gamma)
    loss = focal_term * ce_loss * valid_mask

    # 对于被忽略的像素，其 loss 为 0
    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    # 计算损失时只考虑有效位置
    if reduction == "mean":
        loss = loss.sum() / (valid_mask.sum().clamp_min(eps))
    elif reduction == "sum":
        loss = loss.sum()

    return loss


# -------------------------------
# Dice Loss for Binary Classification
# -------------------------------
class DiceLoss(nn.Module):
    """
    二分类任务的 Dice Loss。
    对 logits 先进行 sigmoid 激活，再计算 Dice 系数。
    支持忽略标签（例如 ignore_index=255）。
    """

    def __init__(self, smooth: float = 0.05, eps: float = 1e-7, ignore_index: int = None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # 如果 target 缺少通道维度，则 unsqueeze 到 [N, 1, H, W]
        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits)

        # 构造有效 mask
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(targets)

        # 展平为 (N, H*W)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        valid_mask = valid_mask.view(valid_mask.size(0), -1)

        # 只计算有效区域
        probs = probs * valid_mask
        targets = targets * valid_mask

        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        dice_loss = 1.0 - dice_score

        # 对每个样本求均值
        return dice_loss.mean()


# -------------------------------
# Combined Loss: Focal + Dice
# -------------------------------
class CombinedBinaryLoss(nn.Module):
    """
    带有辅助损失的二分类损失函数（Focal Loss + Dice Loss）。
    主损失和辅助损失的计算方式相同，辅助损失的权重为 30%。
    """

    def __init__(
            self,
            focal_weight: float = 1.0,
            dice_weight: float = 1.0,
            aux_weight: float = 0.3,  # 辅助损失的权重
            gamma: float = 2.0,
            alpha: float = 0.25,
            reduction: str = "mean",
            ignore_index: int = None,
    ):
        super(CombinedBinaryLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.dice_loss_fn = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor, aux_logits: Tensor = None) -> Tensor:
        # 计算主损失
        focal_loss = focal_loss_with_logits(
            logits, targets, gamma=self.gamma, alpha=self.alpha, reduction=self.reduction,
            ignore_index=self.ignore_index
        )
        dice_loss = self.dice_loss_fn(logits, targets)
        main_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss

        # 计算辅助损失（如果提供了辅助输出）
        if aux_logits is not None:
            aux_focal_loss = focal_loss_with_logits(
                aux_logits, targets, gamma=self.gamma, alpha=self.alpha, reduction=self.reduction,
                ignore_index=self.ignore_index
            )
            aux_dice_loss = self.dice_loss_fn(aux_logits, targets)
            aux_loss = self.focal_weight * aux_focal_loss + self.dice_weight * aux_dice_loss
            total_loss = main_loss + self.aux_weight * aux_loss  # 组合损失
        else:
            total_loss = main_loss

        return total_loss


# -------------------------------
# 示例：如何使用 CombinedBinaryLoss
# -------------------------------
if __name__ == "__main__":
    # 模拟随机生成的 logits 和二值目标（假设为图像分割的二分类任务）
    # logits: 形状 [N, 1, H, W]，targets: 形状 [N, H, W]
    logits = torch.randn((2, 1, 16, 16))
    # 随机生成的标签中有一部分设为 ignore_index (例如255)
    targets = torch.randint(0, 2, (2, 16, 16)).float()
    targets[0, :4, :4] = 255  # 假设前4x4区域需要忽略

    loss_fn = CombinedBinaryLoss(focal_weight=1.0, dice_weight=1.0, gamma=2.0, alpha=0.25, ignore_index=255)
    loss_value = loss_fn(logits, targets)
    print("Combined Binary Loss:", loss_value.item())
