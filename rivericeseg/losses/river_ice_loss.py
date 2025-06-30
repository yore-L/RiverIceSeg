import torch
import torch.nn as nn
import torch.nn.functional as F


class RiverLoss(nn.Module):
    def __init__(self, edge_factor=1.0, connect_weight=0.3, orient_weight=0.2):
        super().__init__()
        # 基础损失组件
        self.edge_factor = edge_factor
        self.connect_weight = connect_weight
        self.orient_weight = orient_weight

        # 初始化卷积核
        self.laplacian_kernel = nn.Parameter(
            torch.tensor([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )

        self.sobel_x = nn.Parameter(
            torch.tensor([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )

        self.sobel_y = nn.Parameter(
            torch.tensor([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )

    def get_boundary(self, x):
        """改进的边界检测方法"""
        # 生成软边界
        x = x.unsqueeze(1).float()
        boundary = F.conv2d(x, self.laplacian_kernel.to(x.device), padding=1)
        boundary = torch.sigmoid(boundary * 5)  # 增强边缘响应
        return boundary.squeeze(1)

    def connectivity_loss(self, pred, target):
        """连通性损失"""
        # 二值化处理
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        target_bin = target.float()

        # 计算骨架相似度
        pred_skel = self.skeletonize(pred_bin)
        target_skel = self.skeletonize(target_bin)
        return F.mse_loss(pred_skel, target_skel)

    def orientation_loss(self, pred, target):
        """方向一致性损失"""
        # 计算梯度方向
        pred_grad_x = F.conv2d(pred.unsqueeze(1), self.sobel_x.to(pred.device), padding=1)
        pred_grad_y = F.conv2d(pred.unsqueeze(1), self.sobel_y.to(pred.device), padding=1)
        pred_orient = torch.atan2(pred_grad_y, pred_grad_x)

        target_grad_x = F.conv2d(target.unsqueeze(1), self.sobel_x.to(target.device), padding=1)
        target_grad_y = F.conv2d(target.unsqueeze(1), self.sobel_y.to(target.device), padding=1)
        target_orient = torch.atan2(target_grad_y, target_grad_x)

        # 计算余弦相似度
        cos_sim = torch.cos(pred_orient - target_orient)
        return 1 - cos_sim.mean()

    def skeletonize(self, x):
        """简易骨架提取"""
        # 形态学细化实现（示例）
        pool1 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        pool2 = 1 - F.max_pool2d(1 - x, kernel_size=3, stride=1, padding=1)
        return torch.abs(pool1 - pool2)

    def forward(self, pred, target):
        # 基础交叉熵 + Dice
        ce_loss = F.binary_cross_entropy_with_logits(pred, target)
        pred_prob = torch.sigmoid(pred)
        dice_loss = 1 - (2 * (pred_prob * target).sum() + 1e-5) / (pred_prob.sum() + target.sum() + 1e-5)

        # 边缘损失
        boundary_target = self.get_boundary(target)
        boundary_pred = self.get_boundary(pred_prob)
        edge_loss = F.binary_cross_entropy(boundary_pred, boundary_target)

        # 组合损失
        total_loss = (ce_loss + dice_loss
                      + self.edge_factor * edge_loss
                      + self.connect_weight * self.connectivity_loss(pred, target)
                      + self.orient_weight * self.orientation_loss(pred, target))

        return total_loss


# 测试用例
if __name__ == "__main__":
    # 模拟输入 (batch=2, H=256, W=256)
    pred = torch.randn(2, 256, 256)
    target = torch.randint(0, 2, (2, 256, 256)).float()

    criterion = RiverLoss()
    loss = criterion(pred, target)
    print(f"Total loss: {loss.item():.4f}")

    # 极端情况测试
    print("Testing edge cases:")
    # 全背景
    bg_target = torch.zeros(2, 256, 256)
    bg_loss = criterion(torch.randn(2, 256, 256), bg_target)
    print(f"All background loss: {bg_loss.item():.4f}")

    # 全前景
    fg_target = torch.ones(2, 256, 256)
    fg_loss = criterion(torch.randn(2, 256, 256), fg_target)
    print(f"All foreground loss: {fg_loss.item():.4f}")
