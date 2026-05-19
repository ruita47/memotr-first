#!/bin/bash

# MeMOTR数据准备脚本
# 用于在AutoDL平台上准备MOT17和CrowdHuman数据集

echo "=== 准备MeMOTR训练数据 ==="

# 创建数据目录
echo "创建数据目录结构..."
mkdir -p /root/autodl-tmp/data/MOT17/images/train
mkdir -p /root/autodl-tmp/data/MOT17/gts/train
mkdir -p /root/autodl-tmp/data/CrowdHuman/images/train
mkdir -p /root/autodl-tmp/data/CrowdHuman/gts/train

# 下载MOT17数据集（如果不存在）
echo "检查MOT17数据集..."
if [ ! -d "/root/autodl-tmp/data/MOT17/images/train/MOT17-02" ]; then
    echo "请手动下载MOT17数据集并解压到 /root/autodl-tmp/data/MOT17/images/ 目录"
    echo "下载链接: https://motchallenge.net/data/MOT17/"
    echo "解压命令示例: unzip MOT17.zip -d /root/autodl-tmp/data/MOT17/images/"
else
    echo "MOT17数据集已存在"
fi

# 下载CrowdHuman数据集（如果不存在）
echo "检查CrowdHuman数据集..."
if [ ! -d "/root/autodl-tmp/data/CrowdHuman/images/train" ]; then
    echo "请手动下载CrowdHuman数据集并解压到 /root/autodl-tmp/data/CrowdHuman/images/ 目录"
    echo "下载链接: https://www.crowdhuman.org/"
    echo "解压命令示例: unzip CrowdHuman_train.zip -d /root/autodl-tmp/data/CrowdHuman/images/train/"
else
    echo "CrowdHuman数据集已存在"
fi

# 生成ground truth文件
echo "生成MOT17 ground truth文件..."
if [ -d "/root/autodl-tmp/data/MOT17/images/train" ]; then
    python data/gen_mot17_gts.py --data-root /root/autodl-tmp/data
else
    echo "MOT17数据集不存在，跳过GT生成"
fi

echo "生成CrowdHuman ground truth文件..."
if [ -d "/root/autodl-tmp/data/CrowdHuman/images/train" ]; then
    python data/gen_crowdhuman_gts.py --data-root /root/autodl-tmp/data
else
    echo "CrowdHuman数据集不存在，跳过GT生成"
fi

echo "数据准备完成！"
echo "数据集结构:"
echo "/root/autodl-tmp/data/"
echo "├── MOT17/"
echo "│   ├── images/"
echo "│   │   ├── train/"
echo "│   │   └── test/"
echo "│   └── gts/"
echo "│       └── train/"
echo "└── CrowdHuman/"
echo "    ├── images/"
echo "    │   ├── train/"
echo "    │   └── val/"
echo "    └── gts/"
echo "        ├── train/"
echo "        └── val/"