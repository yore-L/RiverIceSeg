from .dice import DiceLossV2
from .focal_cosine import FocalLoss
from .edge import EdgeLoss
from .oriConnect import OrientationLoss, ConnectivityLoss

import torch
import torch.nn as nn
from torch import Tensor





class IceLoss(nn.Module):
    def __init__(self,focal_alpha=0.25, focal_gamma=2.0, edge_weight=3.0,connect_weight=0.3,
                 orient_weight=0.2, dice_smooth=0.05, aux_weight=0.3, ignore_index=None):
        super().__init__()
        # 基础参数设置
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.edge_weight = edge_weight
        self.connect_weight = connect_weight
        self.orient_weight = orient_weight
        self.dice_smooth = dice_smooth
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

        # 初始化卷积核
        self.register_buffer('sobel_x',
            torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
            torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

        # 初始化损失函数
        self.focal_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=ignore_index)
        self.dice_fn = DiceLossV2(smooth=dice_smooth, ignore_index=ignore_index)
        self.edge_fn = EdgeLoss(edge_weight=edge_weight, ignore_index=ignore_index)
        self.connect_fn = ConnectivityLoss()
        self.orient_fn = OrientationLoss()

    def forward(self, pred: torch.Tensor, target: Tensor, aux_pred: torch.Tensor = None) -> Tensor:
        # 统一维度为 [Batch, Channel=1, H, W]
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 确保 target 是 float 类型
        target = target.float()

        # 主损失
        focal = self.focal_fn(pred, target)
        dice = self.dice_fn(pred, target)
        edge = self.edge_fn(pred, target)
        connect = self.connect_fn(pred, target)
        orient = self.orient_fn(pred, target)
        main_loss = focal + dice + edge * self.edge_weight + connect * self.connect_weight + orient * self.orient_weight

        # 辅助损失
        if aux_pred is not None:
            if aux_pred.dim() == 3:
                aux_pred = aux_pred.unsqueeze(1)
            focal_a = self.focal_fn(aux_pred, target)
            dice_a = self.dice_fn(aux_pred, target)
            edge_a = self.edge_fn(aux_pred, target)
            connect_a = self.connect_fn(aux_pred, target)
            orient_a = self.orient_fn(aux_pred, target)
            aux_loss = focal_a + dice_a + edge_a * self.edge_weight + connect_a * self.connect_weight + orient_a * self.orient_weight
            main_loss = main_loss + self.aux_weight * aux_loss

        return main_loss

# 测试
if __name__ == "__main__":
    print("Testing ERiverLoss...")
    N, H, W = 3, 1024, 1024
    pred = torch.randn((N, H, W))
    aux_pred = torch.randn((N, H, W))
    targets = torch.randint(0, 2, (N, H, W)).float()  # 确保 targets 是 float 类型
    enhanced_loss_fn = IceLoss(focal_alpha=0.25,
                                  focal_gamma=2.0,
                                  edge_weight=3.0,
                                  connect_weight=0.3,
                                  orient_weight=0.2,
                                  dice_smooth=0.05,
                                  aux_weight=0.3,
                                  ignore_index=255)
    # 主分支损失
    loss_main = enhanced_loss_fn(pred, targets)
    # 带辅助分支损失
    loss_with_aux = enhanced_loss_fn(pred, targets, aux_pred=aux_pred)
    print(f"ERiverLoss (main): {loss_main.item():.4f}")
    print(f"ERiverLoss (with aux): {loss_with_aux.item():.4f}")



