#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeMOTR训练监控脚本
用于在AutoDL平台上监控训练进度和指标
"""

import os
import time
import json
import subprocess
from datetime import datetime

def monitor_training():
    """监控训练进度"""
    
    log_dir = "./outputs/memotr_mot17_autodl/train/"
    
    print("=== MeMOTR训练监控 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"监控目录: {log_dir}")
    
    # 检查GPU状态
    try:
        gpu_info = subprocess.check_output(['nvidia-smi'], text=True)
        print("\n=== GPU状态 ===")
        print(gpu_info.split('\n')[0:15])  # 显示前15行GPU信息
    except Exception as e:
        print(f"无法获取GPU信息: {e}")
    
    # 监控训练日志
    last_iteration = 0
    while True:
        try:
            # 检查训练日志文件
            log_files = []
            if os.path.exists(log_dir):
                log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            
            if log_files:
                latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
                log_path = os.path.join(log_dir, latest_log)
                
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    
                # 解析最新进度
                for line in reversed(lines):
                    if 'iteration' in line and 'loss' in line:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 最新进度: {line.strip()}")
                        break
                    
                    if 'HOTA' in line or 'MOTA' in line or 'ASSA' in line:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 评估指标: {line.strip()}")
                        break
            
            # 检查GPU使用情况（每5分钟）
            if int(time.time()) % 300 == 0:
                try:
                    gpu_usage = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv'], 
                        text=True
                    )
                    print(f"\n=== GPU使用情况 ===")
                    print(gpu_usage)
                except Exception as e:
                    print(f"获取GPU使用情况失败: {e}")
            
            time.sleep(60)  # 每分钟检查一次
            
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"监控错误: {e}")
            time.sleep(60)

def check_training_status():
    """检查训练状态"""
    
    print("\n=== 训练状态检查 ===")
    
    # 检查进程
    try:
        processes = subprocess.check_output(['ps', 'aux'], text=True)
        python_processes = [p for p in processes.split('\n') if 'python' in p and 'main.py' in p]
        
        if python_processes:
            print("训练进程运行中:")
            for p in python_processes:
                print(f"  {p}")
        else:
            print("未发现训练进程")
    except Exception as e:
        print(f"检查进程失败: {e}")
    
    # 检查日志文件
    log_dir = "./outputs/memotr_mot17_autodl/train/"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if log_files:
            latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
            log_mtime = os.path.getmtime(os.path.join(log_dir, latest_log))
            log_age = time.time() - log_mtime
            
            print(f"\n最新日志文件: {latest_log}")
            print(f"最后修改时间: {datetime.fromtimestamp(log_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"距现在: {int(log_age/60)} 分钟前")
            
            if log_age > 600:  # 10分钟没有更新
                print("⚠️  警告: 日志文件长时间未更新，训练可能已停止")
            else:
                print("✅ 训练正常进行中")
        else:
            print("未发现日志文件")
    else:
        print("日志目录不存在")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_training_status()
    else:
        print("使用方法:")
        print("  python train_monitor.py        # 启动实时监控")
        print("  python train_monitor.py status # 检查训练状态")
        
        if len(sys.argv) == 1:
            monitor_training()