=== MeMOTR AutoDL平台快速启动指南 ===

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
