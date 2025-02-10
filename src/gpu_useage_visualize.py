import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import os
import time

# 定义日志文件路径
continuous_log_file = 'gpu_usage_continuous_log.txt'
event_log_file = 'gpu_usage_log.txt'

# 读取持续监控日志
def read_continuous_log(file_path):
    timestamps = []
    memory_used = []
    gpu_loads = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            try:
                parts = line.strip().split(', ')
                timestamp_str = parts[0].split('Timestamp: ')[1]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                mem_used_str = parts[3].split('Memory Used: ')[1].split(' MB')[0]
                mem_used = float(mem_used_str)
                gpu_load_str = parts[5].split('GPU Load: ')[1].strip('%')
                gpu_load = float(gpu_load_str)
                timestamps.append(timestamp)
                memory_used.append(mem_used)
                gpu_loads.append(gpu_load)
            except Exception as e:
                print(f"Error parsing line: {line}")
                continue
    print(f"Continuous log loaded: {len(timestamps)} data points.")
    return pd.DataFrame({'Timestamp': timestamps, 'Memory Used': memory_used, 'GPU Load': gpu_loads})

# 读取事件日志
def read_event_log(file_path):
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            try:
                parts = line.strip().split(', ')
                timestamp_str = parts[0].split('Timestamp: ')[1]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                event_str = parts[1].split('Event: ')[1]
                mem_used_str = parts[4].split('Memory Used: ')[1].split(' MB')[0]
                mem_used = float(mem_used_str)
                events.append({'Timestamp': timestamp, 'Event': event_str, 'Memory Used': mem_used})
            except Exception as e:
                print(f"Error parsing line: {line}")
                continue
    print(f"Event log loaded: {len(events)} events.")
    return pd.DataFrame(events)

# 绘制函数
def plot_memory_used(continuous_df, event_df, event_color_map, save_path):
    # Memory Used 独立图表
    plt.figure(figsize=(12, 6))
    plt.plot(continuous_df['Timestamp'], continuous_df['Memory Used'], label='Memory Used (MB)', color='blue')
    for event_type in event_color_map:
        event_times = event_df[event_df['Event'] == event_type]['Timestamp']
        plt.scatter(event_times, [continuous_df['Memory Used'].max() * 1.05] * len(event_times),
                    label=event_type, marker='v', color=event_color_map[event_type])
    plt.xlabel('Time')
    plt.ylabel('Memory Used (MB)')
    plt.title('GPU Memory Used Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/memory_used_over_time.png")
    plt.close()

def plot_gpu_load(continuous_df, event_df, event_color_map, save_path):
    # GPU Load 独立图表
    plt.figure(figsize=(12, 6))
    plt.plot(continuous_df['Timestamp'], continuous_df['GPU Load'], label='GPU Load (%)', color='green')
    for event_type in event_color_map:
        event_times = event_df[event_df['Event'] == event_type]['Timestamp']
        plt.scatter(event_times, [continuous_df['GPU Load'].max() * 1.05] * len(event_times),
                    label=event_type, marker='v', color=event_color_map[event_type])
    plt.xlabel('Time')
    plt.ylabel('GPU Load (%)')
    plt.title('GPU Load Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/gpu_load_over_time.png")
    plt.close()

def plot_combined(continuous_df, event_df, event_color_map, save_path):
    # GPU Load 和 Memory Used 结合图表
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_memory = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Memory Used (MB)', color=color_memory)
    ax1.plot(continuous_df['Timestamp'], continuous_df['Memory Used'], color=color_memory, label='Memory Used (MB)')
    ax1.tick_params(axis='y', labelcolor=color_memory)

    ax2 = ax1.twinx()  # 共享x轴
    color_load = 'tab:green'
    ax2.set_ylabel('GPU Load (%)', color=color_load)
    ax2.plot(continuous_df['Timestamp'], continuous_df['GPU Load'], color=color_load, label='GPU Load (%)')
    ax2.tick_params(axis='y', labelcolor=color_load)

    # 添加事件标记
    for event_type in event_color_map:
        event_times = event_df[event_df['Event'] == event_type]['Timestamp']
        ax1.scatter(event_times, [continuous_df['Memory Used'].max() * 1.05] * len(event_times),
                    label=event_type, marker='v', color=event_color_map[event_type])

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title('GPU Load and Memory Used Over Time')
    plt.tight_layout()
    plt.savefig(f"{save_path}/combined_over_time.png")
    plt.close()

# 主函数
def continuous_plotting(save_path):
    # 时间控制器
    last_memory_update = time.time()
    last_load_update = time.time()
    last_combined_update = time.time()
    last_event_check = time.time()

    # 设置更新间隔
    memory_interval = 10      # Memory Used 图表更新频率（秒）
    load_interval = 15        # GPU Load 图表更新频率（秒）
    combined_interval = 20    # 结合图表更新频率（秒）
    event_check_interval = 2  # 事件检查频率（秒）

    # 初始事件数据
    last_event_count = 0

    try:
        while True:
            current_time = time.time()

            # 读取数据
            continuous_df = read_continuous_log(continuous_log_file)
            event_df = read_event_log(event_log_file)

            # 事件类型和颜色映射
            event_types = event_df['Event'].unique()
            event_colors = sns.color_palette('hsv', len(event_types))
            event_color_map = dict(zip(event_types, event_colors))

            # 检查是否有新事件
            if len(event_df) > last_event_count or (current_time - last_event_check >= event_check_interval):
                last_event_count = len(event_df)
                last_event_check = current_time
                # 立即更新所有图表
                plot_memory_used(continuous_df, event_df, event_color_map, save_path)
                plot_gpu_load(continuous_df, event_df, event_color_map, save_path)
                plot_combined(continuous_df, event_df, event_color_map, save_path)
                print("All charts updated due to new event.")

            # 更新 Memory Used 图表
            if current_time - last_memory_update >= memory_interval:
                last_memory_update = current_time
                plot_memory_used(continuous_df, event_df, event_color_map, save_path)
                print("Memory Used chart updated.")

            # 更新 GPU Load 图表
            if current_time - last_load_update >= load_interval:
                last_load_update = current_time
                plot_gpu_load(continuous_df, event_df, event_color_map, save_path)
                print("GPU Load chart updated.")

            # 更新结合图表
            if current_time - last_combined_update >= combined_interval:
                last_combined_update = current_time
                plot_combined(continuous_df, event_df, event_color_map, save_path)
                print("Combined chart updated.")

            # 等待一段时间（例如 1 秒）后再次检查
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nUser interrupted the program. Exiting...")

if __name__ == '__main__':
    # 指定保存图片的路径
    save_path = "/root/autodl-tmp/RAG_Agent/src/gpu_monitor"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    continuous_plotting(save_path)
