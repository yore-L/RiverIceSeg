import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ConnectivityLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
        target_bin = target.float()
        pool1 = F.max_pool2d(pred_bin, 3, 1, 1)
        pool2 = 1 - F.max_pool2d(1 - pred_bin, 3, 1, 1)
        pred_skel = torch.abs(pool1 - pool2)
        pool1 = F.max_pool2d(target_bin, 3, 1, 1)
        pool2 = 1 - F.max_pool2d(1 - target_bin, 3, 1, 1)
        target_skel = torch.abs(pool1 - pool2)
        return F.mse_loss(pred_skel, target_skel)

class OrientationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, pred: torch.Tensor, target: Tensor) -> Tensor:
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        pred_orient = torch.atan2(pred_grad_y, pred_grad_x)
        target_orient = torch.atan2(target_grad_y, target_grad_x)
        return 1 - torch.cos(pred_orient - target_orient).mean()