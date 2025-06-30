import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights


class PyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(PyramidPool, self).__init__()
        self.pool_size = max(pool_size, 2)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape
        output = F.interpolate(self.features(x), size=size[2:], mode='bilinear', align_corners=True)
        return output


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PSPNet, self).__init__()
        print("initializing model")
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        self.layer5a = PyramidPool(2048, 512, 1)
        self.layer5b = PyramidPool(2048, 512, 2)
        self.layer5c = PyramidPool(2048, 512, 3)
        self.layer5d = PyramidPool(2048, 512, 6)

        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(512, num_classes, 1),
        )

        initialize_weights(self.layer5a, self.layer5b, self.layer5c, self.layer5d, self.final)

    def forward(self, x):
        size = x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.final(torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1))

        return F.interpolate(x, size=size[2:], mode='bilinear', align_corners=True)


# if __name__ == "__main__":
#     # 随机生成输入数据
#     rgb = torch.randn(1, 3, 1024, 1024)
#     # 定义网络
#     net = PSPNet(num_classes=2, pretrained=False)
#     # 切换到评估模式（避免 BatchNorm 报错）
#     net.eval()
#     # 前向传播
#     out_cls = net(rgb)
#     # 打印输出大小
#     print(out_cls.shape)

if __name__ == '__main__':
    model = PSPNet(num_classes=2, pretrained=False)
    model.eval()

    # 维度验证测试
    def test_dimensions():
        print("\n=== 维度验证测试 ===")
        try:
            x = torch.randn(4, 3, 1024, 1024)
            output = model(x)
            assert output.shape == (4, 2, 1024, 1024), f"维度错误！实际输出维度：{output.shape}"
            print("✅ 维度验证通过 (1024x1024输入)")

            # 测试非标准尺寸输入
            x = torch.randn(4, 3, 512, 512)
            output = model(x)
            assert output.shape == (4, 2, 512, 512), "非标准尺寸验证失败"
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
                x = torch.randn(4, 3, 1024, 1024).to(device)
                output = model(x)
                print(f"✅ {device.upper()} 设备测试通过")
            except Exception as e:
                print(f"❌ {device.upper()} 设备测试失败：", str(e))


    # 梯度检查
    def test_gradient_flow():
        print("\n=== 梯度流检查 ===")
        try:
            model.train()
            device = next(model.parameters()).device

            # 使用兼容的输入尺寸和batch_size=2
            x = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)

            output = model(x)
            loss = output.mean()
            loss.backward()

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

        # 强制评估模式
        model.eval()

        # CPU测试
        model.cpu()
        x = torch.randn(1, 3, 1024, 1024)
        timer = Timer(
            stmt='model(x)',
            globals={'model': model, 'x': x},
            num_threads=torch.get_num_threads()
        )
        print(f"CPU推理耗时：{timer.timeit(10).mean * 1e3:.1f}ms")

        # GPU测试
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