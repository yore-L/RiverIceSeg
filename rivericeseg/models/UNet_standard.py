import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

if __name__ == '__main__':
    # 基础测试
    model = UNet(n_channels=3, n_classes=2, bilinear=True)


    # 维度验证测试
    def test_dimensions():
        print("\n=== 维度验证测试 ===")
        try:
            x = torch.randn(2, 3, 1024, 1024)
            output = model(x)
            assert output.shape == (2, 2, 1024, 1024), f"维度错误！实际输出维度：{output.shape}"
            print("✅ 维度验证通过 (1024x1024输入)")

            # 测试非标准尺寸输入
            x = torch.randn(1, 3, 512, 512)
            output = model(x)
            assert output.shape == (1, 2, 512, 512), "非标准尺寸验证失败"
            print("✅ 非标准尺寸验证通过 (512x512输入)")
        except Exception as e:
            print("❌ 测试失败:", str(e))


    # 参数统计
    def print_model_summary():
        print("\n=== 模型参数统计 ===")
        try:
            from torchsummary import summary
            summary(model.cuda(), (3, 1024, 1024))
        except ImportError:
            print("请先安装torchsummary：pip install torchsummary")

        # 替代方案
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量：{total_params / 1e6:.2f}M")
        print(f"可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")


    # 设备兼容性测试
    def test_device_compatibility():
        print("\n=== 设备兼容性测试 ===")
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        for device in devices:
            try:
                model.to(device)
                x = torch.randn(2, 3, 1024, 1024).to(device)
                output = model(x)
                print(f"✅ {device.upper()} 设备测试通过")
            except Exception as e:
                print(f"❌ {device.upper()} 设备测试失败：", str(e))


    # 梯度检查
    def test_gradient_flow():
        print("\n=== 梯度流检查 ===")
        try:
            model.train()
            # 自动匹配设备
            device = next(model.parameters()).device

            # 创建带梯度的输入
            x = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)

            output = model(x)
            loss = output.mean()
            loss.backward()

            # 检查梯度
            zero_grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"⚠️ 参数无梯度：{name}")
                elif torch.all(param.grad == 0):
                    zero_grad_count += 1

            print(f"✅ 梯度反向传播完成 | 零梯度参数比例：{zero_grad_count / len(list(model.parameters())):.1%}")

        except Exception as e:
            print("❌ 梯度测试失败:", str(e))


    # 运行所有测试
    test_dimensions()
    print_model_summary()
    test_device_compatibility()
    test_gradient_flow()


    # 附加基准测试（可选）
    def benchmark():
        print("\n=== 性能基准测试 ===")
        from torch.utils.benchmark import Timer

        model.cpu()
        x = torch.randn(1, 3, 1024, 1024)
        timer = Timer(
            stmt='model(x)',
            globals={'model': model, 'x': x},
            num_threads=torch.get_num_threads()
        )

        print(f"CPU推理耗时：{timer.timeit(10).mean * 1e3:.1f}ms")

        if torch.cuda.is_available():
            model.cuda()
            x = x.cuda()
            torch.cuda.synchronize()
            timer = Timer(
                stmt='model(x); torch.cuda.synchronize()',
                globals={'model': model, 'x': x}
            )
            print(f"GPU推理耗时：{timer.timeit(10).mean * 1e3:.1f}ms")


    benchmark()