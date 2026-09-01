## Version 1.0(stable)
### Tips
*KAN-UNetFormer*的主要创新点在于**KANConv**。*KANConv*是基于Kolmogorov–Arnold网络（[KAN(Kolmogorov–Arnold Networks)](https://arxiv.org/abs/2404.19756)）创新设计的卷积模块，其核心思想是将KAN的非线性函数逼近特性与卷积操作的局部特征提取能力相结合，在保留传统卷积空间感知优势的同时，通过KAN的全局非线性建模能力增强特征的表征效果。本文基于[UNetformer](https://github.com/WangLibo1995/GeoSeg?tab=readme-ov-file)（Wang等，2021），针对其GLTB模块的局部分支结构，将原有的传统卷积层替换为该KANConv模块，并引入KAN网络的理论框架，重点面向河冰提取任务开展优化研究。

## 介绍
**RivericeSeg**是一个基于 **PyTorch**、 [pytorch lightning](https://lightning.ai/)和[timm](https://github.com/huggingface/pytorch-image-models)的开源语义分割工具箱，主要用于遥感图像分割。

## 主要特点
由于基于UNetformer，故也继承了UNetformer的主要特点
* 统一基准  
    为各种分割方法提供了统一的训练脚本
* 简单有效  
    得益于pytorch lightning和timm，代码很容易进一步开发。
* 支持遥感数据集  
    具体参考[UNetformer](https://github.com/WangLibo1995/GeoSeg?tab=readme-ov-file)
* 多尺度训练和测试

## 支持的网络
*理论上与UNetformer相同*

* Vision TransFormer  
   * [UNetFormer](https://www.sciencedirect.com/science/article/abs/pii/S0924271622001654?via%3Dihub)  
   * [DC-Swin](https://ieeexplore.ieee.org/abstract/document/9681903)  
   * [Swin-Transformer]()
   * [SegFormer]()
   * [KAN-UNetFormer]()

* CNN
    * [UNet]()  
    * [ResNet]()  
    * [ResUnet]()  
    * [PSPNet]()  
    * [SegNet]()  
    * [DeepLab V3+]()

## 文件结构
准备以下文件夹
```
ice
├── RivericeSeg (code)
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── river_ice
│   │   ├── Train
│   │   │   ├── river_ice
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
```

## 安装
*Windows系统也可以，只是开启多线程较为麻烦*   
**下面在Ubuntu 22.04进行安装测试**，使用 Linux Terminal 打开文件夹 ice 并创建 python 环境：
```
conda create -n ice python=3.8  
conda activate ice  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
pip install -r RiverIceSeg/requirements.txt
```

## 预训练权重
[百度网盘](https://pan.baidu.com/share/init?surl=foJkxeUZwVi5SnKNpn6hfg): 1234  
[Google Drive](https://drive.google.com/drive/folders/1ELpFKONJZbXmwB5WCXG7w42eHtrXzyPn)  

## 数据预处理
[Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)、[UAViD](https://uavid.nl/)、[LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)、[OpenEarthMap](https://open-earth-map.org/)可参考[此处](https://github.com/WangLibo1995/GeoSeg?tab=readme-ov-file),这里只介绍自己的数据集。

```
python tools/mask_convert.py --mask-dir data/Train/river_ice/masks_png --output-mask-dir data/Train/river_ice/masks_png_convert  
python tools/mask_convert.py --mask-dir data/Val/river_ice/masks_png --output-mask-dir data/Val/river_ice/masks_png_convert
```

## 训练
"-c" 表示 config 的路径，使用不同的 config 来训练不同的模型。  
```
python ./train_supervision.py -c ./config/river_ice/unetformer.py
```

## 测试

"-c" 表示配置的路径，使用不同的配置来测试不同的模型。   
  
"-o" 表示输出路径   
  
"-t" 表示测试时间增强 （TTA），可以是 [None， 'lr'， 'd4']，默认为 None，'lr' 是翻转 TTA，'d4' 是多尺度 TTA  
  
"--rgb" 表示是否以 RGB 格式输出掩码  
  
```
python ./test.py -c ./config/unetformer.py -o fig_results/output_test -t 'd4' --rgb
```


***由于训练阶段的一些随机操作，复现结果（运行一次）与论文中报告的略有不同。***  

## 引用
如果您在研究中使用本项目，请考虑引用：  
[A KAN–UNetFormer framework for river ice extent extraction from landsat satellite imagery on the Tibetan Plateau](https://www.sciencedirect.com/science/article/abs/pii/S0165232X2600279X?via%3Dihub)

## 致谢
我们希望借助 RiverIceSeg 提供的统一基准来激励研究人员开发自己的分割网络来服务于日益增长的遥感研究。非常感谢以下项目的贡献。  
* [GeoSeg](https://github.com/WangLibo1995/GeoSeg?tab=readme-ov-file)
* [UNetFormer](https://www.sciencedirect.com/science/article/abs/pii/S0924271622001654?via%3Dihub)
* [KAN(Kolmogorov–Arnold Networks)](https://arxiv.org/abs/2404.19756)
* [Kansformers](https://github.com/akaashdash/kansformers)  
* [pytorch lightning](https://lightning.ai/)  
* [timm](https://github.com/huggingface/pytorch-image-models)  
* [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)  
* [ttach](https://github.com/qubvel/ttach)  
* [catalyst](https://github.com/catalyst-team/catalyst)
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
