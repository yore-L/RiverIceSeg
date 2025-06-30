import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)



class ResUnet(nn.Module):
    def __init__(self, in_channel=3, num_classes=2, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output




if __name__ == '__main__':
    # 基础测试
    model = ResUnet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    # 维度验证测试
    def test_dimensions():
        print("\n=== 维度验证测试 ===")
        try:
            x = torch.randn(8, 3, 1024, 1024)
            output = model(x)
            assert output.shape == (8, 2, 1024, 1024), f"维度错误！实际输出维度：{output.shape}"
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