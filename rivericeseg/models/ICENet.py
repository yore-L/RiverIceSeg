import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class TempaeratureScale(nn.Module):
    def __init__(self):
        super(TempaeratureScale, self).__init__()
        self.temp = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        return x / self.temp


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3):
        super(ConvBlock, self).__init__()
        padding = filter_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, filter_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, filter_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class UNetBatchNorm(nn.Module):
    def __init__(self, in_channels=1, n_forecast_days=1, n_forecast_factor=1, filter_size=3, legacy_rounding=False):
        super(UNetBatchNorm, self).__init__()

        start_out_channels = 64
        reduced_channels = start_out_channels * n_forecast_factor
        if not legacy_rounding:
            reduced_channels = int(reduced_channels)

        def calc_channels(pow):
            val = reduced_channels * (2 ** pow)
            return  int(val) if legacy_rounding else val

        self.conv1 = ConvBlock(in_channels, calc_channels(0), filter_size=filter_size)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(calc_channels(0), calc_channels(1), filter_size=filter_size)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(calc_channels(1), calc_channels(2), filter_size=filter_size)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(calc_channels(2), calc_channels(2), filter_size=filter_size)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = ConvBlock(calc_channels(2), calc_channels(3), filter_size=filter_size)

        self.up6 = nn.Conv2d(calc_channels(3), calc_channels(2), kernel_size=2)
        self.conv6 = ConvBlock(calc_channels(2)*2, calc_channels(2), filter_size=filter_size)

        self.up7 = nn.Conv2d(calc_channels(2), calc_channels(2), kernel_size=2)
        self.conv7 = ConvBlock(calc_channels(2)*2, calc_channels(2), filter_size=filter_size)

        self.up8 = nn.Conv2d(calc_channels(2), calc_channels(1), kernel_size=2)
        self.conv8 = ConvBlock(calc_channels(1)*2, calc_channels(1), filter_size=filter_size)

        self.up9 = nn.Conv2d(calc_channels(1), calc_channels(0), kernel_size=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(calc_channels(0)*2, calc_channels(0), filter_size, padding=filter_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(calc_channels(0), calc_channels(0), filter_size, padding=filter_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(calc_channels(0), calc_channels(0), filter_size, padding=filter_size // 2),
            nn.ReLU(inplace=True),
        )

        self.final_layer = nn.Conv2d(calc_channels(0), n_forecast_days, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        u6 = F.interpolate(c5, scale_factor=2, mode='nearest')
        u6 = self.up6(u6)
        c6 = self.conv6(torch.cat([c4, u6], dim=1))

        u7 = F.interpolate(c6, scale_factor=2, mode='nearest')
        u7 = self.up7(u7)
        c7 = self.conv7(torch.cat([c3, u7], dim=1))

        u8 = F.interpolate(c7, scale_factor=2, mode='nearest')
        u8 = self.up8(u8)
        c8 = self.conv8(torch.cat([c2, u8], dim=1))

        u9 = F.interpolate(c8, scale_factor=2, mode='nearest')
        u9 = self.up9(u9)
        c9 = self.conv9(torch.cat([c1, u9], dim=1))

        out = self.final_layer(c9)
        return self.activation(out)



def linear_trend_forecast(
    usable_selector: object,
    forecast_date: object,
    da: object,
    mask: object,
    missing_dates: object = (),
    shape: object = (432, 432)
) -> object:
    """

    :param usable_selector:
    :param forecast_date:
    :param da:
    :param mask:
    :param missing_dates:
    :param shape:
    :return:
    """
    usable_data = usable_selector(da, forecast_date, missing_dates)

    if len(usable_data.time) < 1:
        return np.full(shape, np.nan)

    x = np.arange(len(usable_data.time))
    y = usable_data.data.reshape(len(usable_data.time), -1)

    src = np.c_[x, np.ones_like(x)]
    r = np.linalg.lstsq(src, y, rcond=None)[0]
    output_map = np.matmul(np.array([len(usable_data.time), 1]),
                           r).reshape(*shape)
    output_map[mask] = 0.
    output_map[output_map < 0] = 0.
    output_map[output_map > 1] = 1.

    return output_map



# 假设模型是 UNetBatchNorm，已经定义好
model = UNetBatchNorm(in_channels=3, n_forecast_days=1)
model.eval()

# 模拟输入图像：[1, 3, 1024, 1024]
input_image = torch.rand(1, 3, 1024, 1024)

# 模拟真实标签：二分类标签图（0 和 1）
true_mask = torch.randint(0, 2, (1, 1, 1024, 1024)).float()

# 模型推理
with torch.no_grad():
    pred = model(input_image)
    pred_binary = (pred > 0.5).float()

# 计算指标
def compute_metrics(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    TP = ((pred == 1) & (target == 1)).sum().item()
    TN = ((pred == 0) & (target == 0)).sum().item()
    FP = ((pred == 1) & (target == 0)).sum().item()
    FN = ((pred == 0) & (target == 1)).sum().item()

    eps = 1e-7  # 防止除0

    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)

    return accuracy, precision, recall, f1, iou

acc, prec, rec, f1, iou = compute_metrics(pred_binary, true_mask)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"IoU: {iou:.4f}")