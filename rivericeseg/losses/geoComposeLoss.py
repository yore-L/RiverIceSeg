from .dice import *
from .edge import *
from .oriConnect import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructEdgeLoss(nn.Module):
    def __init__(self,
                 edge_factor: float = 1.0,
                 connect_weight: float = 0.3,
                 orient_weight: float = 0.2,
                 aux_edge_weight: float = 0.3,  # 辅助边缘权重
                 ignore_index: int = None):
        super().__init__()
        self.edge_factor = edge_factor
        self.connect_weight = connect_weight
        self.orient_weight = orient_weight
        self.aux_edge_weight = aux_edge_weight
        self.ignore_index = ignore_index
        self.lap = nn.Parameter(torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
                                 .view(1, 1, 3, 3), requires_grad=False)
        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                                     .view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
                                     .view(1, 1, 3, 3), requires_grad=False)
        self.ce_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_fn = DiceLossV2(smooth=0.05, ignore_index=ignore_index)
        self.edge_fn = EdgeLoss(edge_weight=3.0, ignore_index=ignore_index)
        self.conn_fn = ConnectivityLoss()
        self.orient_fn = OrientationLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_logits: torch.Tensor = None) -> torch.Tensor:
        # 统一维度为 [Batch, Channel=1, H, W]
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # 确保 target 是 float 类型
        targets = targets.float()

        # 主损失
        ce_map = self.ce_fn(logits, targets)
        dice = self.dice_fn(logits, targets)

        # 主边缘监督
        bd_pred = torch.sigmoid(F.conv2d(torch.sigmoid(logits), self.lap, padding=1) * 5).squeeze(1)
        bd_tgt = torch.sigmoid(F.conv2d(targets, self.lap, padding=1) * 5).squeeze(1)
        edge = F.binary_cross_entropy(bd_pred, bd_tgt)

        # 辅助边缘监督
        aux_edge = 0.0
        if aux_logits is not None:
            if aux_logits.dim() == 3:
                aux_logits = aux_logits.unsqueeze(1)
            bd_aux = torch.sigmoid(F.conv2d(torch.sigmoid(aux_logits), self.lap, padding=1) * 5).squeeze(1)
            aux_edge = F.binary_cross_entropy(bd_aux, bd_tgt)

        # 连通性
        connect = self.conn_fn(logits, targets)

        # 方向
        orient = self.orient_fn(logits, targets)

        # 总损失
        main_loss = ce_map.mean() + dice + self.edge_factor * edge + \
                    self.aux_edge_weight * aux_edge + \
                    self.connect_weight * connect + \
                    self.orient_weight * orient

        return main_loss

