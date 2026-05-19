#!/bin/bash

# AutoDL平台启动脚本
# 用于在AutoDL平台上启动MeMOTR训练

echo "=== MeMOTR AutoDL训练启动 ==="

# 设置环境变量
export PYTHONPATH=$PWD:$PYTHONPATH

# 检查CUDA可用性
echo "检查CUDA设备..."
nvidia-smi
echo "CUDA版本:" && nvcc --version

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 编译Deformable Attention CUDA ops
echo "编译Deformable Attention CUDA ops..."
cd ./models/ops/
chmod +x make.sh
./make.sh

# 测试CUDA ops
echo "测试CUDA ops..."
python test.py

cd ../..

# 创建数据目录结构
echo "创建数据目录结构..."
mkdir -p /root/autodl-tmp/data/MOT17/images/train
mkdir -p /root/autodl-tmp/data/MOT17/gts/train
mkdir -p /root/autodl-tmp/data/CrowdHuman/images/train
mkdir -p /root/autodl-tmp/data/CrowdHuman/gts/train

# 下载预训练模型（如果需要）
echo "下载预训练模型..."
if [ ! -f "dab_deformable_detr.pth" ]; then
    wget -O dab_deformable_detr.pth "https://drive.google.com/uc?export=download&id=17FxIGgIZJih8LWkGdlIOe9ZpVZ9IRxSj"
fi

# 修改配置文件以适应AutoDL环境
echo "配置训练参数..."
cp configs/train_mot17.yaml configs/train_mot17_autodl.yaml

# 更新配置参数
sed -i 's/DATA_ROOT:.*/DATA_ROOT: \/root\/autodl-tmp\/data/' configs/train_mot17_autodl.yaml
sed -i 's/OUTPUTS_DIR:.*/OUTPUTS_DIR: .\/outputs\/memotr_mot17_autodl/' configs/train_mot17_autodl.yaml
sed -i 's/USE_CHECKPOINT:.*/USE_CHECKPOINT: True/' configs/train_mot17_autodl.yaml
sed -i 's/BATCH_SIZE:.*/BATCH_SIZE: 1/' configs/train_mot17_autodl.yaml
sed -i 's/ACCUMULATION_STEPS:.*/ACCUMULATION_STEPS: 4/' configs/train_mot17_autodl.yaml

# 添加长时记忆增强参数
echo "MAX_KEYFRAMES: 30" >> configs/train_mot17_autodl.yaml
echo "KEYFRAME_SIM_THRESHOLD: 0.9" >> configs/train_mot17_autodl.yaml
echo "KEYFRAME_CONF_THRESHOLD: 0.5" >> configs/train_mot17_autodl.yaml
echo "LONG_MEMORY_LAMBDA: 0.01" >> configs/train_mot17_autodl.yaml

echo "配置完成！"
echo "=== 开始训练 ==="

# 启动训练（单GPU版本，适合AutoDL环境）
python main.py \
    --mode train \
    --config-path ./configs/train_mot17_autodl.yaml \
    --outputs-dir ./outputs/memotr_mot17_autodl/ \
    --batch-size 1 \
    --data-root /root/autodl-tmp/data \
    --use-checkpoint \
    --device cuda

echo "训练完成！"