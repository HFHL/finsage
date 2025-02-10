from datetime import datetime
import GPUtil
import os

# 定义全局日志文件路径
log_file = os.path.join('/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/logs','gpu_usage_log.txt')
gpu_log_file = log_file

# Author: hhl
# Date: 2024/10/22
# Description: 记录gpu使用情况
def log_gpu_usage(event_name):
    gpus = GPUtil.getGPUs()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for gpu in gpus:
        gpu_info = (
            f"Timestamp: {timestamp}, Event: {event_name}, "
            f"GPU ID: {gpu.id}, GPU Name: {gpu.name}, "
            f"Memory Used: {gpu.memoryUsed} MB, Memory Total: {gpu.memoryTotal} MB"
        )
        # 将信息追加到日志文件
        with open(gpu_log_file, 'a') as f:
            f.write(gpu_info + '\n')
