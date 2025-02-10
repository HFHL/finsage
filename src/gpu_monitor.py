import GPUtil
import time
import datetime

def monitor_gpu_usage(log_file='gpu_usage_continuous_log.txt', interval=1):
    f = open(log_file, 'w')
    while True:
        gpus = GPUtil.getGPUs()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for gpu in gpus:
            gpu_info = (
                f"Timestamp: {timestamp}, "
                f"GPU ID: {gpu.id}, GPU Name: {gpu.name}, "
                f"Memory Used: {gpu.memoryUsed} MB, Memory Total: {gpu.memoryTotal} MB, "
                f"GPU Load: {gpu.load * 100:.1f}%"
            )
            f.write(gpu_info + '\n')
        time.sleep(interval)

if __name__ == '__main__':
    monitor_gpu_usage(interval=0.5)