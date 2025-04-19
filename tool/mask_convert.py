import glob
import os
import numpy as np
import cv2
from osgeo import gdal
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import random

SEED = 42

CLASSES = ('background', 'ice')

PALETTE = [[0, 0, 0], [255, 255, 255]]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="data/LoveDA/Train/Rural/masks_png")
    parser.add_argument("--output-mask-dir", default="data/LoveDA/Train/Rural/masks_png_convert")
    return parser.parse_args()


def convert_label(mask):
    mask[mask == 0] = 0
    # mask -= 1
    mask[mask == 1] = 1

    return mask


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 255, 255]

    return mask_rgb


def read_tif_image(image_path):
    """使用 gdal 读取遥感影像"""
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open image file {image_path}")

    # 假设图像是多波段的，提取第一个波段（通常是RGB或单波段）
    band = dataset.GetRasterBand(1)
    image = band.ReadAsArray()
    return image


def patch_format(inp):
    (mask_path, masks_output_dir) = inp
    # print(mask_path, masks_output_dir)
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    mask = read_tif_image(mask_path)
    label = convert_label(mask)
    rgb_label = label2rgb(label.copy())
    rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)
    out_mask_path_rgb = os.path.join(masks_output_dir + '_rgb', "{}.tif".format(mask_filename))
    cv2.imwrite(out_mask_path_rgb, rgb_label)

    out_mask_path = os.path.join(masks_output_dir, "{}.tif".format(mask_filename))
    cv2.imwrite(out_mask_path, label)


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    masks_dir = args.mask_dir
    masks_output_dir = args.output_mask_dir
    mask_paths = glob.glob(os.path.join(masks_dir, "*.tif"))

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        os.makedirs(masks_output_dir + '_rgb')

    inp = [(mask_path, masks_output_dir) for mask_path in mask_paths]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


