#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeMOTR项目上传到AutoDL平台的脚本
"""

import os
from pathlib import Path

def create_upload_script():
    """创建上传脚本"""
    
    script_content = """#!/bin/bash

# MeMOTR项目上传脚本
# 用于将项目上传到AutoDL平台

echo "=== 上传MeMOTR项目到AutoDL ==="

# AutoDL连接信息
AUTODL_HOST="connect.nmb2.seetacloud.com"
AUTODL_PORT=47325
AUTODL_USER="root"

# 本地项目路径（请修改为您的实际路径）
LOCAL_PROJECT="/path/to/your/MeMOTR-main"

# 远程项目路径
REMOTE_PROJECT="/root/MeMOTR-main"

echo "连接信息:"
echo "主机: $AUTODL_HOST"
echo "端口: $AUTODL_PORT"
echo "用户: $AUTODL_USER"
echo "本地项目: $LOCAL_PROJECT"
echo "远程项目: $REMOTE_PROJECT"

echo ""
echo "请确保本地项目路径正确，然后按Enter继续..."
read

# 使用rsync同步项目文件
echo "开始同步项目文件..."
rsync -avz -e "ssh -p $AUTODL_PORT" \\
    --exclude='.git' \\
    --exclude='outputs/' \\
    --exclude='__pycache__/' \\
    --exclude='*.pyc' \\
    --exclude='.DS_Store' \\
    $LOCAL_PROJECT/ $AUTODL_USER@$AUTODL_HOST:$REMOTE_PROJECT/

echo "文件同步完成！"

echo ""
echo "下一步操作:"
echo "1. 连接到AutoDL服务器: ssh -p $AUTODL_PORT $AUTODL_USER@$AUTODL_HOST"
echo "2. 进入项目目录: cd $REMOTE_PROJECT"
echo "3. 准备数据: bash prepare_data.sh"
echo "4. 启动训练: bash autodl_start.sh"
echo "5. 监控训练: python train_monitor.py"
"""
    
    with open("upload_script.sh", "w") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod("upload_script.sh", 0o755)
    
    print("上传脚本已创建: upload_script.sh")

def create_ssh_config():
    """创建SSH配置文件"""
    
    config_content = """Host autodl-memotr
    HostName connect.nmb2.seetacloud.com
    Port 47325
    User root
    ServerAliveInterval 60
    ServerAliveCountMax 3
"""
    
    with open("ssh_config.txt", "w") as f:
        f.write(config_content)
    
    print("SSH配置示例已创建: ssh_config.txt")

def generate_quick_start_guide():
    """生成快速启动指南"""
    
    guide = """=== MeMOTR AutoDL平台快速启动指南 ===

## 1. 上传项目到AutoDL平台

### 方法1: 使用rsync（推荐）
```bash
# 修改upload_script.sh中的LOCAL_PROJECT路径
vim upload_script.sh

# 运行上传脚本
bash upload_script.sh
```

### 方法2: 手动上传
```bash
# 压缩项目
tar -czf MeMOTR-main.tar.gz MeMOTR-main/

# 上传到AutoDL
scp -P 47325 MeMOTR-main.tar.gz root@connect.nmb2.seetacloud.com:/root/

# 在AutoDL服务器上解压
ssh -p 47325 root@connect.nmb2.seetacloud.com
cd /root
tar -xzf MeMOTR-main.tar.gz
```

## 2. 连接到AutoDL服务器

```bash
ssh -p 47325 root@connect.nmb2.seetacloud.com
```

## 3. 在AutoDL服务器上执行

```bash
# 进入项目目录
cd /root/MeMOTR-main

# 安装依赖
pip install -r requirements.txt

# 编译CUDA ops
cd models/ops/
chmod +x make.sh
./make.sh
python test.py
cd ../..

# 准备数据（需要先下载数据集）
bash prepare_data.sh

# 启动训练
bash autodl_start.sh

# 监控训练进度（新终端）
python train_monitor.py
```

## 4. 训练参数配置

- 使用GPU进行训练
- 每100次迭代返回一次损失
- 每个epoch返回HOTA, MOTA, ASSA等指标
- 长时记忆增强机制已启用
- 关键帧存储策略已配置

## 5. 关键配置参数

- MAX_KEYFRAMES: 30
- KEYFRAME_SIM_THRESHOLD: 0.9
- KEYFRAME_CONF_THRESHOLD: 0.5
- LONG_MEMORY_LAMBDA: 0.01

## 6. 数据集: MOT17

## 注意事项

- 确保MOT17和CrowdHuman数据集已下载到/root/autodl-tmp/data/
- 训练过程会占用较多GPU内存，已启用梯度检查点优化
- 训练时间预计需要数小时到数天，具体取决于GPU性能
- 可以通过tensorboard查看训练曲线: tensorboard --logdir outputs/memotr_mot17_autodl/train/

## 监控训练

### 实时监控
```bash
python train_monitor.py
```

### 检查状态
```bash
python train_monitor.py status
```

### TensorBoard可视化
```bash
tensorboard --logdir outputs/memotr_mot17_autodl/train/ --port 6006
```
"""
    
    with open("AUTODL_QUICK_START.md", "w") as f:
        f.write(guide)
    
    print("快速启动指南已创建: AUTODL_QUICK_START.md")

def create_direct_connect_script():
    """创建直接连接脚本"""
    
    script_content = """#!/bin/bash

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
python main.py \\
    --mode train \\
    --config-path ./configs/train_mot17.yaml \\
    --outputs-dir ./outputs/memotr_mot17_autodl/ \\
    --batch-size 1 \\
    --data-root /root/autodl-tmp/data \\
    --use-checkpoint \\
    --device cuda

echo "训练已启动!"

EOF

echo "连接已建立，训练正在后台运行"
echo "可以使用以下命令监控训练:"
echo "ssh -p 47325 root@connect.nmb2.seetacloud.com 'cd /root/MeMOTR-main && python train_monitor.py status'"
"""
    
    with open("direct_connect.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("direct_connect.sh", 0o755)
    print("直接连接脚本已创建: direct_connect.sh")

if __name__ == "__main__":
    print("=== 准备MeMOTR项目上传到AutoDL平台 ===")
    
    # 创建必要的文件
    create_upload_script()
    create_ssh_config()
    generate_quick_start_guide()
    create_direct_connect_script()
    
    print("\n=== 上传准备完成 ===")
    print("已创建以下文件:")
    print("- upload_script.sh: 项目上传脚本")
    print("- ssh_config.txt: SSH配置示例")
    print("- direct_connect.sh: 直接连接训练脚本")
    print("- AUTODL_QUICK_START.md: 详细使用指南")
    print("\n请按照AUTODL_QUICK_START.md中的说明操作")