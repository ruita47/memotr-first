#!/bin/bash

# MeMOTR AutoDL直接连接和训练脚本

echo "=== MeMOTR AutoDL直接连接 ==="

# 连接到AutoDL服务器并启动训练
ssh -p 47325 root@connect.nmb2.seetacloud.com << 'EOF'

# 在远程服务器上执行的命令
echo "=== 在AutoDL服务器上启动MeMOTR训练 ==="

# 检查项目目录
if [ ! -d "/root/MeMOTR-main" ]; then
    echo "错误: MeMOTR项目目录不存在"
    echo "请先上传项目文件"
    exit 1
fi

cd /root/MeMOTR-main

# 检查依赖
echo "检查Python依赖..."
pip list | grep -E "(torch|torchvision|opencv)" || echo "需要安装依赖"

# 检查CUDA ops
echo "检查CUDA ops..."
cd models/ops/
if [ ! -f "build/lib.linux-x86_64-cpython-310/ms_deform_attn.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "编译CUDA ops..."
    chmod +x make.sh
    ./make.sh
    python test.py
fi
cd ../..

# 检查数据
echo "检查数据集..."
if [ ! -d "/root/autodl-tmp/data/MOT17/images/train" ]; then
    echo "警告: MOT17数据集不存在"
    echo "请先下载数据集到 /root/autodl-tmp/data/"
fi

# 启动训练
echo "启动MeMOTR训练..."
python main.py \
    --mode train \
    --config-path ./configs/train_mot17.yaml \
    --outputs-dir ./outputs/memotr_mot17_autodl/ \
    --batch-size 1 \
    --data-root /root/autodl-tmp/data \
    --use-checkpoint \
    --device cuda

echo "训练已启动!"

EOF

echo "连接已建立，训练正在后台运行"
echo "可以使用以下命令监控训练:"
echo "ssh -p 47325 root@connect.nmb2.seetacloud.com 'cd /root/MeMOTR-main && python train_monitor.py status'"
