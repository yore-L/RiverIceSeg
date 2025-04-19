import torch
from torch import nn
from torch.nn import functional as F

import cv2


class EdgeLoss(nn.Module):
    """二分类边界损失，通过形态学梯度检测边缘并加权 BCE，支持 ignore_index"""

    def __init__(self, edge_weight: float = 3.0, kernel_size: int = 3, ignore_index: int = None):
        super().__init__()
        self.edge_weight = edge_weight
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index

    def get_edge_mask(self, targets: torch.Tensor) -> torch.Tensor:
        edge_masks = []
        for i in range(targets.shape[0]):
            mask = targets[i].squeeze().cpu().numpy().astype('uint8')
            if self.ignore_index is not None:
                mask[mask == self.ignore_index] = 0
            kern = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
            dilated = cv2.dilate(mask, kern)
            eroded = cv2.erode(mask, kern)
            edge = (dilated - eroded).astype(bool)
            edge_tensor = torch.from_numpy(edge.astype('float32')).to(targets.device)
            edge_masks.append(edge_tensor)
        edge_mask = torch.stack(edge_masks).unsqueeze(1)
        return edge_mask * self.edge_weight + 1.0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        edge_mask = self.get_edge_mask(targets)
        loss = F.binary_cross_entropy_with_logits(logits, targets, weight=edge_mask, reduction='none')
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            loss = loss * mask
            return loss.sum() / mask.sum().clamp_min(1e-6)
        return loss.mean()
