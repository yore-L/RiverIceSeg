import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tool.cfg import py2cfg
import os
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
from tool.metric import Evaluator
from tool.plotter import Plotter, RealtimePlotCallback
from pytorch_lightning.loggers import CSVLogger
import random

CUDA_LAUNCH_BLOCKING=1

deterministic = True  # 设置为True表示使用确定性训练，False表示不使用
# print(f"CPU 核心数: {os.cpu_count()}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if not deterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.metric_train = Evaluator(num_class=config.num_classes)
        self.metric_val = Evaluator(num_class=config.num_classes)
        self.total_loss = 0.0
        self.num_batches = 0

    def forward(self, x):
        # 仅在预测/推理中使用网络
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        
        # 检查预测结果的形状并进行适当调整，确保与损失函数的预期输入匹配
        if isinstance(prediction, tuple) and self.config.use_aux_loss:
            # 处理带有辅助输出的情况
            main_pred, aux_pred = prediction
            
            if main_pred.shape[1] == 2:  # 二分类情形，选择第二个通道（通常是前景概率）
                # 将形状从[N,2,H,W]变为[N,1,H,W]
                loss_pred = main_pred[:, 1:2, :, :]
                aux_loss_pred = aux_pred[:, 1:2, :, :] if aux_pred.shape[1] == 2 else aux_pred
                loss = self.loss((loss_pred, aux_loss_pred), mask)
            # else:
            #     # 直接使用预测结果
            #     loss = self.loss(main_pred, mask, aux_pred)
        else:
            # 处理没有辅助输出的情况
            if prediction.shape[1] == 2:  # 二分类情形，选择第二个通道（前景）
                loss_pred = prediction[:, 1:2, :, :]
                loss = self.loss(loss_pred, mask)
            else:
                loss = self.loss(prediction, mask)
            # print(loss)
        # 后处理用于指标计算的预测结果
        if self.config.use_aux_loss and isinstance(prediction, tuple):
            pre_mask = torch.sigmoid(prediction[0])
        else:
            pre_mask = torch.sigmoid(prediction)
        
        # 如果预测有两个通道，仅使用第二个通道（前景）进行评估
        if pre_mask.shape[1] == 2:
            pre_mask = pre_mask[:, 1:2, :, :]
        
        pre_mask = pre_mask.squeeze(1)
        # 使用阈值0.5进行二分类
        pre_mask = (pre_mask > 0.5).long()  # 将概率转为类别标签（0或1）

        for i in range(mask.shape[0]):
            self.metric_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # print('loss:',loss.item())
        self.total_loss += loss.item()
        self.num_batches += 1

        return {"loss": loss}

    def on_train_epoch_end(self):
        # 计算损失
        avg_loss = self.total_loss / self.num_batches if self.num_batches > 0 else 0.0
        print(f'Average train Loss: {avg_loss}')
        if 'river_ice' in self.config.log_name:
            mIoU = np.nanmean(self.metric_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metric_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metric_train.Intersection_over_Union())
            F1 = np.nanmean(self.metric_train.F1())
        OA = np.nanmean(self.metric_train.OA())
        Dice = np.nanmean(self.metric_train.Dice())
        Recall = np.nanmean(self.metric_train.Recall())
        iou_per_class = self.metric_train.Intersection_over_Union()
        eval_value = {
            'mIoU': mIoU,
            'F1': F1,
            'OA': OA,
            'Dice': Dice,
            'Recall': Recall,
        }
        print('train: ',eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        # print(iou_value)

        self.metric_train.reset()
        log_dict = {'train_river_ice': iou_per_class[1], 'train_mIoU': mIoU, 'train_F1': F1, 'train_Dice': Dice, 'train_Recall': Recall, 'train_OA': OA, 'train_loss': avg_loss}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        
        # 与训练步骤类似，调整预测输出以匹配损失函数预期
        if isinstance(prediction, tuple) and self.config.use_aux_loss:
            main_pred, aux_pred = prediction
            
            if main_pred.shape[1] == 2:
                loss_pred = main_pred[:, 1:2, :, :]
                aux_loss_pred = aux_pred[:, 1:2, :, :] if aux_pred.shape[1] == 2 else aux_pred
                loss_val = self.loss((loss_pred, aux_loss_pred), mask)
            else:
                loss_val = self.loss(main_pred, mask, aux_pred)
        else:
            if prediction.shape[1] == 2:
                loss_pred = prediction[:, 1:2, :, :]
                loss_val = self.loss(loss_pred, mask)
            else:
                loss_val = self.loss(prediction, mask)
        
        # 后处理用于指标计算的预测结果
        if isinstance(prediction, tuple) and self.config.use_aux_loss:
            pre_mask = torch.sigmoid(prediction[0])
        else:
            pre_mask = torch.sigmoid(prediction)
        
        # 如果预测有两个通道，仅使用第二个通道（前景）进行评估
        if pre_mask.shape[1] == 2:
            pre_mask = pre_mask[:, 1:2, :, :]
        
        pre_mask = pre_mask.squeeze(1)
        pre_mask = (pre_mask > 0.5).long()  # 根据阈值0.5将概率转为类标签（0或1）

        for i in range(mask.shape[0]):
            # 直接使用处理好的预测结果和标签
            pred = pre_mask[i].cpu().numpy()
            gt = mask[i].cpu().numpy()
            
            # # 打印调试信息
            # if i == 0 and batch_idx == 0:
            #     print(f"预测形状: {pred.shape}, 标签形状: {gt.shape}")
            self.metric_val.add_batch(gt, pred)
            # print('val_loss:',loss_val.item())
            self.total_loss += loss_val.item()
            self.num_batches += 1

        return {'val_loss': loss_val}

    def on_validation_epoch_end(self):
        # 计算损失
        avg_loss = self.total_loss / self.num_batches if self.num_batches > 0 else 0.0
        print(f'Average Val Loss: {avg_loss}')
        if 'river_ice' in self.config.log_name:
            mIoU = np.nanmean(self.metric_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metric_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metric_val.Intersection_over_Union())
            F1 = np.nanmean(self.metric_val.F1())
        OA = np.nanmean(self.metric_val.OA())
        Dice = np.nanmean(self.metric_val.Dice())
        Recall = np.nanmean(self.metric_val.Recall())
        iou_per_class = self.metric_val.Intersection_over_Union()

        eval_value = {
            'mIoU': mIoU,
            'F1': F1,
            'OA': OA,
            'Dice': Dice,
            'Recall': Recall,
        }
        print('val: ', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        # print(iou_value)

        self.metric_val.reset()
        log_dict = {'val_river_ice': iou_per_class[1], 'val_mIoU': mIoU, 'val_F1': F1, 'val_Dice': Dice, 'val_Recall': Recall, 'val_OA': OA, 'val_loss': avg_loss}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        # 添加错误处理，并返回具有安全设置的dataloader
        if hasattr(self.config, 'train_loader') and self.config.train_loader is not None:
            # 确保已存在的dataloader具有适当的设置
            return self.config.train_loader
        else:
            print("警告：未找到train_loader，请检查配置")
            return None

    def val_dataloader(self):
        # 添加错误处理，并返回具有安全设置的dataloader
        if hasattr(self.config, 'val_loader') and self.config.val_loader is not None:
            # 确保已存在的dataloader具有适当的设置
            return self.config.val_loader
        else:
            print("警告：未找到val_loader，请检查配置")
            return None


# 训练
def main():
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Train(config)
    if config.pretrained_ckpt_path:
        model = Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=checkpoint_callback, strategy='auto', logger=logger)


    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)

if __name__ == "__main__":
    main()