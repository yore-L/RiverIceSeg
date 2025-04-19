from torch.utils.data import DataLoader
import sys

from rivericeseg.losses import *
from rivericeseg.datasets.dataset import *
from rivericeseg.models.DeepLabV3plus import DeepLabV3Plus
# from models.UNet_standard import UNet
from tool.utils import Lookahead
from tool.utils import process_model_params


# 训练参数
max_epoch = 500
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 3e-4  # 主网络学习率
weight_decay = 1e-5  # 主网络权重衰减
backbone_lr = 1e-5  # 预训练backbone的学习率要小于主网络
backbone_weight_decay = 0.01  # 预训练backbone的权重衰减要大于主网络
num_classes = len(CLASSES)
classes = CLASSES

weights_name = 'river_ice'
weights_path = 'model_weights/DeepLabV3Plus/{}'.format(weights_name)
test_weights_name = 'last'
log_name = 'river_ice/deeplabv3p{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None    # 预训练模型权重的路径
gpus = 'auto'   # 默认或者gpu ids:[0]或gpu nums:2，更多设置参考pytorch_lightning
resume_ckpt_path = None    # 是否连续


# 定义网络 - 使用ResNet101作为主干网络
net = DeepLabV3Plus(
    n_classes=num_classes,  # 使用与数据集匹配的类别数
    backbone='resnet101',   # 使用预训练的ResNet101
    output_stride=16        # 设置输出步长，影响特征图分辨率
)

# 定义损失函数
loss = CombinedBinaryLoss(ignore_index=ignore_index)
use_aux_loss = False  # DeepLabV3+不支持辅助损失

# 数据加载
def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

train_dataset = TrainDataset(transform=train_aug, data_root='data/train_val')

val_dataset = val_dataset
test_dataset = TestDataset()

# 修改DataLoader配置，避免Windows上的模块导入错误
if sys.platform.startswith('win'):
    # Windows系统上使用安全设置
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=0,  # 使用0个工作进程，避免多进程导入问题
                              pin_memory=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=0,  # 使用0个工作进程，避免多进程导入问题
                            pin_memory=True,
                            drop_last=False)
else:
    # 非Windows系统上使用原始设置
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=4,  # 根据您的CPU核心数调整
                              pin_memory=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=4,  # 根据您的CPU核心数调整
                            pin_memory=True,
                            drop_last=False)


# 优化器 - 使用不同的学习率设置不同层
# 为预训练的backbone网络使用较小的学习率，促进特征提取能力的保留
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)

# 使用AdamW优化器 - 对权重衰减有更好的处理
base_optimizer = torch.optim.AdamW(
    net_params, 
    lr=lr, 
    weight_decay=weight_decay,
    betas=(0.9, 0.999)  # 使用推荐的动量参数
)

# 使用Lookahead优化器进一步提升收敛性能
try:
    optimizer = Lookahead(base_optimizer)
except Exception as e:
    print(f"警告: Lookahead优化器初始化失败: {str(e)}，使用基础优化器代替")
    optimizer = base_optimizer

# 使用余弦退火学习率调度器
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=max_epoch, 
    eta_min=1e-6  # 设置最小学习率
)

