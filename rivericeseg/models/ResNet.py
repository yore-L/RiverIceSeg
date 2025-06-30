import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # 搭建BasicBlock模块
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 使用BN层不需要bias，bias最后会抵消掉
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        Y = self.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))

        if self.downsample is not None:    # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(x)

        return self.relu(Y + identity)


class BottleNeck(nn.Module):
    """搭建BottleNeck模块"""
    # BottleNeck模块最终输出out_channel是Residual模块输入in_channel的size的4倍(Residual模块输入为64)，shortcut分支in_channel
    # 为Residual的输入64，因此需要在shortcut分支上将Residual模块的in_channel扩张4倍，使之与原始输入图片X的size一致
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()

        # 默认原始输入为256，经过7x7层和3x3层之后BottleNeck的输入降至64
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)    # Residual中第三层out_channel扩张到in_channel的4倍

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        Y = self.relu(self.bn1(self.conv1(x)))
        Y = self.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(Y + identity)


class ResNet(nn.Module):
    """搭建ResNet-layer通用框架"""
    # num_classes是训练集的分类个数，include_top是在ResNet的基础上搭建更加复杂的网络时用到，此处用不到
    def __init__(self, residual, num_residual, num_classes=2, include_top=True):
        super(ResNet, self).__init__()
        self.out_channels = 64
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, self.out_channels, kernel_size=7, stride=2, padding=3, bias=False)    # 3表示输入特征图像的RGB通道数为3，即图片数据的输入通道为3
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.residual_block(residual, 64, num_residual[0])
        self.conv3 = self.residual_block(residual, 128, num_residual[1], stride=2)
        self.conv4 = self.residual_block(residual, 256, num_residual[2], stride=2)
        self.conv5 = self.residual_block(residual, 512, num_residual[3], stride=2)
        # if self.include_top:
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #     self.fc = nn.Linear(512 * residual.expansion, num_classes)

        # Decoder部分（使用转置卷积进行上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 * residual.expansion, 256, kernel_size=3, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        )

        # 对conv层进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(nn.BatchNorm2d, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def residual_block(self, residual, channels, num_residuals, stride=1):
        downsample = None
        # 用在每个conv_x组块的第一层的shortcut分支上，此时上个conv_x输出out_channel与本conv_x所要求的输入in_channel通道数不同，
        # 所以用downsample调整进行升维，使输出out_channel调整到本conv_x后续处理所要求的维度。
        # 同时stride=2进行下采样减小尺寸size，(注：conv2时没有进行下采样，conv3-5进行下采样，size=56、28、14、7)。
        if stride != 1 or self.out_channels != channels * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channels, channels * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * residual.expansion),
            )
        block = []  # block列表保存某个conv_x组块里for循环生成的所有层
        # 添加每一个conv_x组块里的第一层，第一层决定此组块是否需要下采样(后续层不需要)
        block.append(residual(self.out_channels, channels, downsample=downsample, stride=stride))
        self.out_channels = channels * residual.expansion

        for _ in range(1, num_residuals):
            block.append(residual(self.out_channels, channels))

        # 非关键字参数的特征是一个星号*加上参数名，比如*number，定义后，number可以接收任意数量的参数，并将它们储存在一个tuple中
        return nn.Sequential(*block)

    def forward(self, x):
        Y = self.relu(self.bn1(self.conv1(x)))
        Y = self.maxpool(Y)
        Y = self.conv5(self.conv4(self.conv3(self.conv2(Y))))

        Y = self.decoder(Y)
        Y = F.interpolate(Y, size=x.size()[2:], mode='bilinear', align_corners=False)

        # if self.include_top:
        #     Y = self.avgpool(Y)
        #     Y = torch.flatten(Y, 1)
        #     Y = self.fc(Y)

        return Y


# 构建ResNet-34
def ResNet34(num_classes=2, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, include_top)

# 构建ResNet-50
def ResNet50(num_classes=2, include_top=True):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, include_top)


if __name__ == '__main__':
    # 基础测试
    model = ResNet34(num_classes=2, include_top=True)


    # 维度验证测试
    def test_dimensions():
        print("\n=== 维度验证测试 ===")
        try:
            x = torch.randn(1, 3, 1024, 1024)
            output = model(x)
            assert output.shape == (1, 2, 1024, 1024), f"维度错误！实际输出维度：{output.shape}"
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = ResNet34(num_classes=2, include_top=True).to(device)
            summary(model, (3, 1024, 1024))
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
                x = torch.randn(1, 3, 1024, 1024).to(device)
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
            x = torch.randn(1, 3, 256, 256, device=device, requires_grad=True)

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