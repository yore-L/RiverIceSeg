import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, resnet50

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ASPPModule, self).__init__()
        self.branches = nn.ModuleList()
        self.branches.append(
            # image pooling 分支
            nn.Sequential(
                nn.AvgPool2d(3, 1, 1),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        # 四个空洞卷积分支
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, dilation=d, padding=d),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 1×1卷积
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d((len(dilations)+1) * out_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.size()[2:]
        # print('size:', size)
        features = []
        # 获取各分支的特征，并把大小调整到一致
        for i in range(len(self.branches)):
            out = self.branches[i](x)
            # print("out.shape:", out.shape)
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
            # print("upsample out.shape:", out.shape)
            features.append(out)
        # 按通道维度合并五个特征分支
        feature = torch.cat(features, dim=1)
        return self.conv_bn_relu(feature)


# 凯明初始化
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


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes=2, backbone='resnet101', output_stride=16):
        super(DeepLabV3Plus, self).__init__()

        # backbone 匹配字典
        backbone_config = {
            'resnet50': {
                'model': resnet50,
                'weight': 'IMAGENET1K_V1',
                'layer_indices': [3, 5, 6, 7]
            },
            'resnet101': {
                'model': resnet101,
                'weight': 'IMAGENET1K_V1',
                'layer_indices': [3, 5, 6, 7]
            }
        }
        if backbone not in backbone_config:
            raise ValueError('Backbone must be one of {}'.format(backbone_config))

        config = backbone_config[backbone]

        if backbone == 'resnet50' or backbone == 'resnet101':
            base_model = config['model'](weights=config['weight'])
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])

            # 分解backbone层
            layer_indices = config['layer_indices']
            self.first = self.backbone[0:layer_indices[0]]
            self.layer1 = self.backbone[layer_indices[0]+1]
            self.layer2 = self.backbone[layer_indices[1]]
            self.layer3 = self.backbone[layer_indices[2]]
            self.layer4 = self.backbone[layer_indices[3]]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use resnet'.format(backbone))

        # if backbone == 'resnet101':
        #     # 这里要用新的写法，否则会显示警告信息，提示过期
        #     # self.backbone = resnet101(pretrained=False)
        #     self.backbone = resnet101(weights='IMAGENET1K_V1')
        #     # 修改ResNet的最后几层以适应DeepLabV3+
        #     # 移除最后的平均池化层和分类层
        #     self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        #     self.first = self.backbone[0:3]
        #     self.layer1 = self.backbone[4]
        #     self.layer2 = self.backbone[5]
        #     self.layer3 = self.backbone[6]
        #     self.layer4 = self.backbone[7]
        # else:
        #     raise ValueError('Unsupported backbone - `{}`, Use resnet'.format(backbone))

        self.aspp = ASPPModule(2048, 256, [1, 6, 12, 18])
        self.conv1x1 = nn.Conv2d(256, 48, 1, 1)
        self.upsample4 = nn.ConvTranspose2d(48, 48, 4, stride=2, padding=1)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(96, n_classes, 3, 1, 1)
        initialize_weights(self.backbone, self.aspp, self.conv1x1, self.upsample4, self.low_level_conv)

    def forward(self, x):
        # 获取主干网络的特征图
        c2, c3, c4, c5 = self._forward_backbone(x)
        size0 = x.size()[2:]
        # print("size0: ", size0)
        # ASPP模块
        features = self.aspp(c5)
        # print("features.shape: ", features.shape)
        features = self.conv1x1(features)
        # print("features.shape: ", features.shape)
        features = self.upsample4(features)
        # print("features.shape: ", features.shape)
        # 低级特征融合
        low_level_features = self.low_level_conv(c3)
        size = low_level_features.size()[2:]
        features = F.interpolate(features, size=size, mode='bilinear', align_corners=True)
        features = torch.cat([features, low_level_features], dim=1)
        # 最终分类层
        output = self.final_conv(features)
        # 最终上采样
        output = F.interpolate(output, size=size0, mode='bilinear', align_corners=True)
        return output

    def _forward_backbone(self, x):
        c2 = self.first(x)
        c3 = self.layer1(c2)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)
        c5 = self.layer4(c5)
        # print("c2.shape: {}".format(c2.shape))
        # print("c3.shape: {}".format(c3.shape))
        # print("c4.shape: {}".format(c4.shape))
        # print("c5.shape: {}".format(c5.shape))
        return c2, c3, c4, c5


if __name__ == '__main__':
    model = DeepLabV3Plus(n_classes=2, backbone='resnet101')

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