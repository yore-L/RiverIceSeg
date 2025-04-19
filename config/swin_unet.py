from torch.utils.data import DataLoader
import sys

from rivericeseg.losses import *
from rivericeseg.datasets.dataset import *
from rivericeseg.models.Swin_UNet import SwinUnet, MockConfig
from tool.utils import Lookahead
from tool.utils import process_model_params


# 训练参数
max_epoch = 500
ignore_index = len(CLASSES)
train_batch_size = 2  # 减小批量大小以适应更大的模型
val_batch_size = 2
lr = 2e-4  # 降低学习率以提高稳定性
weight_decay = 1e-5
num_classes = len(CLASSES)
classes = CLASSES

weights_name = 'river_ice'
weights_path = 'model_weights/SwinUnet/{}'.format(weights_name)
test_weights_name = 'last'
log_name = 'river_ice/swin{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None    # 预训练模型权重的路径
gpus = 'auto'   # 默认或者gpu ids:[0]或gpu nums:2，更多设置参考pytorch_lightning
resume_ckpt_path = None    # 是否连续


# 创建SwinUNet配置
swin_config = MockConfig()
# MockConfig默认就是1024，不需要修改，只需要确保img_size参数保持一致
print(f"配置初始化后的图像尺寸: {swin_config.DATA.IMG_SIZE}")
swin_config.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]  # 减少深度以降低显存占用
swin_config.MODEL.SWIN.WINDOW_SIZE = 8
swin_config.MODEL.DROP_PATH_RATE = 0.1  # 减少丢弃率提高稳定性
swin_config.MODEL.NUM_CLASSES = num_classes

# 定义网络
net = SwinUnet(
    config=swin_config,
    img_size=1024,  # 使用1024作为img_size参数，匹配输入图像尺寸
    num_classes=num_classes,
    zero_head=False
)

# 定义损失函数
loss = CombinedBinaryLoss(ignore_index=ignore_index)
use_aux_loss = False  # SwinUnet不支持辅助损失

# 数据加载
def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # 裁剪尺寸应该与模型期望的输入尺寸匹配 - 使用1024
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25], mode='value'),
                        SmartCropV1(crop_size=1024, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
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
                              num_workers=2,  # 减少工作进程数以减少内存占用
                              pin_memory=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=2,  # 减少工作进程数以减少内存占用
                            pin_memory=True,
                            drop_last=False)


# 优化器
# SwinUNet通常不按照分层调整学习率，所以使用简单的参数组
params = [
    {"params": list(net.parameters()), "lr": lr}
]

# 移除fused参数，提高兼容性
try:
    # 尝试使用融合版本的AdamW (如果支持的话)
    base_optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
except TypeError:
    # 如果不支持fused参数，则使用标准版本
    print("警告: AdamW不支持fused参数，使用标准实现")
    base_optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

# 尝试使用Lookahead优化器，但如果有问题则回退到基础优化器
try:
    optimizer = Lookahead(base_optimizer)
except Exception as e:
    print(f"警告: Lookahead优化器初始化失败: {str(e)}，使用基础优化器代替")
    optimizer = base_optimizer

# 使用更稳定的学习率调度器配置
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

