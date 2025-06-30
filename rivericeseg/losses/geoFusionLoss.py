from .joint_loss import *
from .geoComposeLoss import *

import torch
import torch.nn as nn


class GeoFusionLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        aux_weight: float = 0.3,
        gamma: float = 2.0,
        alpha: float = 0.25,
        edge_factor: float = 1.0,
        connect_weight: float = 0.3,
        orient_weight: float = 0.2,
        aux_edge_weight: float = 0.3,
        lambda_combined: float = 1.0,
        lambda_river: float = 1.0,
        ignore_index: int = None
    ):
        super().__init__()
        self.combined = CombinedBinaryLoss(
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            aux_weight=aux_weight,
            gamma=gamma,
            alpha=alpha,
            ignore_index=ignore_index
        )
        self.river = StructEdgeLoss(
            edge_factor=edge_factor,
            connect_weight=connect_weight,
            orient_weight=orient_weight,
            aux_edge_weight=aux_edge_weight,
            ignore_index=ignore_index
        )
        self.lambda_c = lambda_combined
        self.lambda_r = lambda_river

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_logits: torch.Tensor = None) -> torch.Tensor:
        t = targets.float()
        if t.dim() == logits.dim() - 1:
            t = t.unsqueeze(1)
        loss_c = self.combined(logits, t, aux_logits)
        loss_r = self.river(logits, t.squeeze(1), aux_logits.squeeze(1) if aux_logits is not None else None)
        return self.lambda_c * loss_c + self.lambda_r * loss_r


# Example
if __name__ == "__main__":
    print("Testing HybridRiverLoss...")
    N, H, W = 3, 1024, 1024
    logits = torch.randn((N, 1, H, W))
    targets = torch.randint(0, 2, (N, H, W)).float()
    hybrid_loss_fn = GeoFusionLoss(ignore_index=255)
    loss_hybrid = hybrid_loss_fn(logits, targets)
    print(f"HybridRiverLoss: {loss_hybrid.item():.4f}")

    # 边界测试
    print("Edge case tests for HybridRiverLoss")
    bg_target = torch.zeros((2, H, W))
    fg_target = torch.ones((2, H, W))
    bg_logits = torch.randn((2, 1, H, W))
    fg_logits = torch.randn((2, 1, H, W))
    print("All background loss:", hybrid_loss_fn(bg_logits, bg_target).item())
    print("All foreground loss:", hybrid_loss_fn(fg_logits, fg_target).item())