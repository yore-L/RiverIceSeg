import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import sys
import os
import traceback
import rasterio
import types
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 全局变量声明
has_rasterio = False


def laber2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)

    # 检查并确保掩码是2D数组
    if len(mask.shape) > 2:
        # 如果掩码有多个通道，只使用第一个通道
        mask = mask[:, :, 0] if mask.shape[2] > 0 else mask.squeeze()

    # 为确保安全性，将掩码转换为布尔类型
    mask_0 = (mask == 0)
    mask_1 = (mask > 0)  # 使用 > 0 而不是 == 1，以捕获所有非零值

    # 安全地设置颜色
    mask_rgb[mask_0, :] = [0, 0, 0]  # 背景为黑色
    mask_rgb[mask_1, :] = [255, 255, 255]  # 前景为白色

    return mask_rgb


def img_writer(pred, pred_path, geo_info=None):
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)

        # 确保输出文件扩展名为.tif
        if not pred_path.lower().endswith('.tif'):
            pred_path = pred_path.rsplit('.', 1)[0] + '.tif'

        # 保存方式
        if geo_info and 'crs' in geo_info and has_rasterio:
            # 获取地理信息
            crs = geo_info.get('crs')
            transform = geo_info.get('transform')

            # 确保预测结果是正确的形状和类型
            if len(pred.shape) == 3 and pred.shape[2] == 3:  # RGB格式
                # 使用rasterio保存GeoTIFF
                with rasterio.open(
                        pred_path,
                        'w',
                        driver='GTiff',
                        height=pred.shape[0],
                        width=pred.shape[1],
                        count=3,
                        dtype=pred.dtype,
                        crs=crs,
                        transform=transform
                ) as dst:
                    # RGB顺序写入
                    for i in range(3):
                        dst.write(pred[:, :, i], i + 1)

                    # 复制原始标签
                    if 'tags' in geo_info:
                        dst.update_tags(**geo_info['tags'])
            else:
                # 二值掩码格式
                # 调整为适合rasterio的格式
                if len(pred.shape) == 2:
                    pred_raster = pred  # 单通道
                else:
                    pred_raster = pred.squeeze()  # 确保是2维

                # 使用rasterio保存GeoTIFF
                with rasterio.open(
                        pred_path,
                        'w',
                        driver='GTiff',
                        height=pred_raster.shape[0],
                        width=pred_raster.shape[1],
                        count=1,
                        dtype=pred_raster.dtype,
                        crs=crs,
                        transform=transform
                ) as dst:
                    dst.write(pred_raster, 1)

                    # 复制原始标签
                    if 'tags' in geo_info:
                        dst.update_tags(**geo_info['tags'])
        else:
            # 无地理信息的TIF格式保存
            # 确保预测掩码是正确的格式
            if len(pred.shape) == 3 and pred.shape[2] == 3:  # RGB格式
                # 使用普通方式保存TIF
                cv2.imwrite(pred_path, pred)
            else:
                # 单通道格式
                if len(pred.shape) == 2:
                    # 单通道图像，直接保存
                    cv2.imwrite(pred_path, pred * 255 if pred.dtype == np.float32 or pred.dtype == np.float64 else pred)
                elif len(pred.shape) == 3 and pred.shape[2] == 1:
                    # 单通道但有额外维度，去掉额外维度
                    cv2.imwrite(pred_path,
                                pred.squeeze() * 255 if pred.dtype == np.float32 or pred.dtype == np.float64 else pred.squeeze())
                else:
                    # 多通道但不是RGB
                    # 尝试转换为RGB
                    pred_rgb = np.stack([pred.squeeze()] * 3, axis=2) if len(pred.shape) < 3 else pred
                    cv2.imwrite(pred_path, pred_rgb)

    except Exception as e:
        print(f"保存预测结果时出错: {str(e)}")
        traceback.print_exc()
        # 尝试以最基本的方式保存
        try:
            if len(pred.shape) == 2:
                cv2.imwrite(pred_path, pred * 255 if pred.dtype == np.float32 or pred.dtype == np.float64 else pred)
            else:
                cv2.imwrite(pred_path, pred)
            print("已使用备用方法保存预测结果")
        except Exception as inner_e:
            print(f"备用保存方法也失败了: {str(inner_e)}")

    return 0


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None,
        choices=[None, "d4", "lr"])  ## lr is flip TTA, d4 is multi-scale TTA
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    arg("--weights_path", type=str, default=None, help="Path to model weights")
    arg("--ext", type=str, default=".tif", help="Output file extension (default: .tif)")
    args = parser.parse_args()

    # 确保输出扩展名以点开头
    if args.ext and not args.ext.startswith('.'):
        args.ext = '.' + args.ext

    return args


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    # 如果未指定权重路径，使用配置中的默认路径
    weights_path = args.weights_path if args.weights_path else config.weights_path
    model_path = os.path.join(weights_path, config.test_weights_name + '.ckpt')

    print(f"加载模型：{model_path}")
    model = Train.load_from_checkpoint(model_path, config=config)
    model.cuda()
    model.eval()

    # 优化TTA配置，降低显存占用
    if args.tta == "lr":
        print("使用TTA: 水平+垂直翻转")
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        print("使用TTA: 多尺度增强")
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Scale(scales=[0.75, 1.25], interpolation='bilinear', align_corners=False),
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

        # 设置批量大小
        batch_size_test = 4
    else:
        # 不使用TTA
        batch_size_test = 8

    test_dataset = config.test_dataset
    if args.val:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
        test_dataset = config.val_dataset

    # 修改测试数据集的__getitem__方法，动态添加地理信息加载功能
    if has_rasterio:
        # 保存原始__getitem__方法
        original_getitem = test_dataset.__getitem__

        # 定义新的__getitem__方法
        def getitem_with_geo(self, idx):
            # 获取原始数据
            data = original_getitem(idx)

            # 尝试加载地理信息
            try:
                # 获取原始图像的路径 - 从TestDataset类结构获取路径
                img_id, img_type = self.img_ids[idx]
                img_path = os.path.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)

                if os.path.exists(img_path) and img_path.endswith(('.tif', '.tiff')):
                    with rasterio.open(img_path) as src:
                        # 获取地理信息
                        geo_info = {
                            'crs': src.crs,
                            'transform': src.transform,
                            'tags': src.tags(),
                            'bounds': src.bounds,
                            'path': img_path
                        }
                        data['geo_info'] = geo_info
                else:
                    data['geo_info'] = None
            except Exception as e:
                print(f"加载地理信息失败: {str(e)}")
                data['geo_info'] = None

            return data

        # 替换__getitem__方法
        test_dataset.__getitem__ = types.MethodType(getitem_with_geo, test_dataset)
        print(f"已增强测试数据集以支持地理信息加载")

    with torch.no_grad():
        # 修改DataLoader设置，减少显存占用
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False
        )

        results = []
        batch_size = 1000

        print("开始预测...")
        for input in tqdm(test_loader):
            # 使用.cuda()来移动数据到GPU
            input_img = input['img'].cuda()

            # 记录原始输入尺寸
            original_size = (input_img.shape[2], input_img.shape[3])

            # 检查模型期望的输入尺寸
            expected_size = None

            # 尝试从不同位置获取模型的期望输入尺寸
            if hasattr(config, 'swin_config') and hasattr(config.swin_config.DATA, 'IMG_SIZE'):
                expected_size = config.swin_config.DATA.IMG_SIZE
            elif hasattr(model, 'net') and hasattr(model.net, 'img_size'):
                expected_size = model.net.img_size
            elif hasattr(model, 'net') and hasattr(model.net, 'swin_unet') and hasattr(model.net.swin_unet, 'img_size'):
                expected_size = model.net.swin_unet.img_size

            # 如果检测到尺寸不匹配，调整输入尺寸
            if expected_size is not None and (
                    input_img.shape[2] != expected_size or input_img.shape[3] != expected_size):
                # 调整图像尺寸为模型期望的尺寸
                input_img = torch.nn.functional.interpolate(
                    input_img,
                    size=(expected_size, expected_size),
                    mode='bilinear',
                    align_corners=False
                )

            # 逐步处理预测结果，减少峰值内存占用
            raw_predictions = model(input_img)

            # 立即释放输入图像的GPU内存
            del input_img
            torch.cuda.empty_cache()

            # 计算sigmoid和阈值化，并立即将结果移回CPU
            predictions = nn.Sigmoid()(raw_predictions)

            predictions = (predictions > 0.5).float()  # 根据阈值0.5将概率转换为类标签（0或1）

            # 确保二分类预测只有一个通道
            if predictions.shape[1] == 2:  # 如果有两个通道（背景和前景）
                predictions = predictions[:, 1:2]  # 只保留前景通道

            # 如果之前调整过尺寸，将预测结果调整回原始尺寸
            if expected_size is not None and (original_size[0] != expected_size or original_size[1] != expected_size):
                predictions = torch.nn.functional.interpolate(
                    predictions,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )

            predictions = predictions.cpu()

            # 释放原始预测的GPU内存
            del raw_predictions
            torch.cuda.empty_cache()

            # 处理当前批次的所有样本
            for i in range(predictions.shape[0]):
                mask = predictions[i].numpy()  # 已经在CPU上，直接转numpy

                # 确保输出的掩码是正确的格式：[H, W] 或 [H, W, 1]
                if len(mask.shape) == 3:
                    if mask.shape[0] == 1:
                        mask = mask.squeeze(0)  # 从[1, H, W]转为[H, W]
                    else:
                        # 如果是[C, H, W]格式，转换为[H, W, C]
                        mask = np.transpose(mask, (1, 2, 0))

                mask_name = input["img_id"][i]
                # 处理mask_type可能不存在的情况
                mask_type = input["mask_type"][i] if "mask_type" in input else "default"

                # 准备输出路径和地理信息
                geo_info = input['geo_info'][i] if 'geo_info' in input else None

                # 确保使用正确的文件扩展名
                if not mask_name.lower().endswith(args.ext.lower()):
                    mask_name = mask_name + args.ext

                if args.val:
                    if not os.path.exists(os.path.join(args.output_path, mask_type)):
                        os.makedirs(os.path.join(args.output_path, mask_type), exist_ok=True)
                    evaluator.add_batch(pre_image=mask, gt_image=input['gt_semantic_seg'][i].numpy())

                    # 设置输出路径
                    output_path = str(args.output_path / mask_type / mask_name)
                else:
                    # 设置输出路径
                    output_path = str(args.output_path / mask_name)

                # 如果需要RGB转换
                if args.rgb:
                    # 将掩码转换为二值图像
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    # 转换为RGB
                    mask_rgb = laber2rgb(mask_binary)
                    # 添加结果，不再使用元组
                    results.append((mask_rgb, output_path, geo_info))
                else:
                    # 添加结果，不再使用元组
                    results.append((mask, output_path, geo_info))

            # 释放预测结果的CPU内存
            del predictions

            t0 = time.time()
            ctx = mp.get_context('spawn')
            # 如果积累的结果数量足够多，立即写入磁盘并释放内存
            if len(results) >= batch_size:
                print(f"写入结果...")
                # 使用函数而不是多进程池，以确保img_writer函数按预期工作
                for result in results:
                    # mpp.Pool(processes=ctx.cpu_count()).map_async(img_writer, (result[0], result[1], result[2]))
                    img_writer(result[0], result[1], result[2])
                results = []  # 清空结果列表，释放内存
                # 强制垃圾回收
                gc.collect()

        # 处理剩余不足 batch_size 的结果
        if len(results) > 0:
            print(f"写入剩余结果...")
            # 使用函数而不是多进程池，以确保img_writer函数按预期工作
            for result in results:
                # mpp.Pool(processes=ctx.cpu_count()).map_async(img_writer, (result[0], result[1], result[2]))
                img_writer(result[0], result[1], result[2])
            results = []
            # 强制垃圾回收
            gc.collect()
        t1 = time.time()

        if args.val:
            iou_per_class = evaluator.Intersection_over_Union()
            f1_per_class = evaluator.F1()
            OA = evaluator.OA()
            for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
                print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
            print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))

        img_writer_time = t1 - t0
        print('测试与写入完成，总计处理图像: {}张,用时{}s'.format(len(test_dataset), img_writer_time))


if __name__ == "__main__":
    main()


