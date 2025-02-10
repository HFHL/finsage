import subprocess
import sys
import os
import logging
import json
import threading  # 引入线程锁
import time
import concurrent.futures  # 引入并行执行库

os.sync()  # 强制将内存中的数据写入磁盘

# 配置日志
log_file = '/root/autodl-tmp/file2chunk/script/main_pipeline_2nd.log'
logging.basicConfig(
    filename=log_file, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建一个锁对象
lock = threading.Lock()

def replace_spaces_in_path(path):
    """替换路径中的空格为下划线"""
    return path.replace(" ", "_")

def validate_path(path):
    """增强路径验证"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {path}")
    return os.path.abspath(path)

def build_step_paths(input_path, output_dir):
    """构建各步骤路径的工厂函数"""
    base_name = os.path.basename(input_path)
    base_name = base_name.replace('_content_list', '').replace('.json', '')
    base_name = replace_spaces_in_path(base_name)  # 替换路径中的空格
    
    paths = {
        # 步骤配置
        'step2_output': os.path.join(output_dir, base_name, "base.json"),
        'step3_output': os.path.join(output_dir, base_name, "base_remove_empty.json"),
        'step4_output': os.path.join(output_dir, base_name, "base_remove_empty_similarity_results.json"),
        'step5_output': os.path.join(output_dir, base_name, "base_remove_empty_deduped.json"),
        'step6_output': os.path.join(output_dir, base_name, "base_remove_empty_deduped_processed.json"),  # 修改为processed
        'step7_output': os.path.join(output_dir, base_name, "base_processed_chunked.json"),  # 修改为processed_chunked
        'step8_output': os.path.join(output_dir, base_name, "base_final.json"),  # 最终输出命名为final
        
        # 中间目录
        'work_dir': os.path.join(output_dir, base_name)
    }
    
    return paths

def check_file_existence(file_path):
    """确保文件存在，增加锁机制"""
    with lock:  # 锁住检查文件操作，避免并发读取/写入冲突
        while not os.path.exists(file_path):
            logging.info(f"文件 {file_path} 尚未生成，等待中...")
            time.sleep(1)  # 等待1秒后再次检查

def reset_ids(file_path):
    """重设 JSON 文件中的 ID，从 1 开始"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    current_id = 1
    for item in data:
        if 'id' in item:  # 只处理有 id 字段的项
            item['id'] = current_id
            current_id += 1

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_final_json(step7_output_path, final_output_path):
    """生成 final.json"""
    with open(step7_output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 此处可以根据需求进行进一步的处理
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_document(input_path, output_dir, similarity_threshold, max_workers, enable_summary):
    try:
        # ========== 初始化验证 ==========
        input_path = validate_path(input_path)
        output_dir = validate_path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取各步骤路径
        path_config = build_step_paths(input_path, output_dir)
        work_dir = path_config['work_dir']
        os.makedirs(work_dir, exist_ok=True)

        # ========== Step 2: 基础处理 ==========
        logging.info("[Step 2/6] 启动表格处理流程...")
        step2_cmd = [
            "python", "step2_mineru2base.py",
            input_path,
            output_dir
        ]
        subprocess.run(step2_cmd, check=True)
        
        check_file_existence(path_config['step2_output'])  # 使用锁机制检查文件是否生成
        logging.info(f"表格处理完成: {path_config['step2_output']}")

        # ========== Step 3: 数据清洗 ==========
        logging.info("[Step 3/6] 启动数据清洗...")
        step3_cmd = [
            "python", "step3_remove_empty_content.py",
            path_config['step2_output']
        ]
        subprocess.run(step3_cmd, check=True)
        
        check_file_existence(path_config['step3_output'])  # 使用锁机制检查文件是否生成
        logging.info(f"数据清洗完成: {path_config['step3_output']}")

        # ========== Step 4: 相似度分析 ==========
        logging.info("[Step 4/6] 启动相似度分析...")
        step4_cmd = [
            "python", "step4_similarity_analysis.py",
            path_config['step3_output'],
            "--threshold", str(similarity_threshold),
            "--max_workers", str(max_workers)
        ]
        subprocess.run(step4_cmd, check=True)
        
        check_file_existence(path_config['step4_output'])  # 使用锁机制检查文件是否生成
        logging.info(f"相似度分析完成: {path_config['step4_output']}")

        # ========== Step 5: 去重处理 ==========
        logging.info("[Step 5/6] 启动去重处理...")
        step5_cmd = [
            "python", "step5_delete_similar_chunks.py",
            path_config['step4_output'],  # 相似度结果文件
            path_config['step3_output']   # 待去重的基础文件
        ]
        subprocess.run(step5_cmd, check=True)
        
        check_file_existence(path_config['step5_output'])  # 使用锁机制检查文件是否生成
        logging.info(f"去重处理完成: {path_config['step5_output']}")

        # 例如 Step 5 完成后增加文件验证
        if os.path.exists(path_config['step5_output']):
            logging.info(f"步骤 5 输出文件已生成: {path_config['step5_output']}")
        else:
            logging.error(f"步骤 5 输出文件未生成: {path_config['step5_output']}")

        time.sleep(2)  # 等待5秒，确保文件生成完毕

        # ========== Step 6: 指代消解与摘要生成 ==========
        logging.info("[Step 6/6] 启动指代消解与摘要生成...")

        check_file_existence(path_config['step5_output'])  # 使用锁机制检查文件是否生成
        
        # 检查文件内容是否有效
        try:
            with open(path_config['step5_output'], 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    raise ValueError("步骤5输出文件为空")
        except Exception as e:
            raise ValueError(f"步骤5输出文件无效: {str(e)}")

        # 确保输出目录存在
        step6_output_dir = os.path.dirname(path_config['step6_output'])
        os.makedirs(step6_output_dir, exist_ok=True)

        # 执行步骤6
        step6_cmd = [
            "python", "step6_anaphora_resolution.py",
            os.path.abspath(path_config['step5_output']),  # 使用绝对路径，确保路径正确
            os.path.abspath(path_config['step6_output']),  # 使用绝对路径

            "--generate-summary" if enable_summary else ""
        ]
        subprocess.run(step6_cmd, check=True)

        check_file_existence(path_config['step6_output'])  # 使用锁机制检查文件是否生成
        logging.info(f"指代消解与摘要生成完成: {path_config['step6_output']}")

        # ========== Step 7: 文本分块 ==========
        logging.info("[Step 7/6] 启动文本分块处理...")
        step7_cmd = [
            "python", "step7_split_chunks.py",  # 确保脚本名是正确的
            "-i", path_config['step6_output'],  # 输入文件
            "-o", path_config['step7_output'],  # 输出文件
            "-s", str(256)  # 块大小
        ]
        subprocess.run(step7_cmd, check=True)

        check_file_existence(path_config['step7_output'])  # 使用锁机制检查文件是否生成
        logging.info(f"文本分块处理完成: {path_config['step7_output']}")

        # ========== Step 8: 重设 ID ==========
        logging.info("[Step 8/6] 启动 ID 重设处理...")
        reset_ids(path_config['step7_output'])  # 重设 step7 输出

        # ========== Step 9: 生成 final.json ==========
        logging.info("[Step 9/6] 生成 final.json 文件...")
        generate_final_json(path_config['step7_output'], path_config['step8_output'])  # 生成 final.json
        logging.info(f"final.json 生成完成: {path_config['step8_output']}")

    except subprocess.CalledProcessError as e:
        logging.error(f"子流程执行失败: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"流程错误: {str(e)}")
        sys.exit(1)

def main():
    # 输入文件路径列表
    # input_files = [
    #     "/root/autodl-tmp/file2chunk/mineru2/Lotus F-1 20240503/auto/Lotus F-1 20240503_content_list.json",
    #     "/root/autodl-tmp/file2chunk/mineru2/Lotus 424B3 20241112/auto/Lotus 424B3 20241112_content_list.json"
    # ]
    

    # 设置根目录
    root_folder = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/experiment/processed_pdf/43-84/Base"

    # 存储所有 `_content_list.json` 文件的路径
    content_list_files = []

    # 遍历所有子文件夹
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith("_content_list.json"):
                content_list_files.append(os.path.join(subdir, file))

    # # 打印所有匹配的文件路径
    # for content_file in content_list_files:
    #     print(content_file)
    input_files = content_list_files

    
    output_dir = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/experiment/pipeline_2nd_half"
    similarity_threshold = 0.7
    max_workers = 5
    enable_summary = True

    # 使用多线程处理每个输入文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for input_path in input_files:
            futures.append(executor.submit(process_document, input_path, output_dir, similarity_threshold, max_workers, enable_summary))

        # 等待所有线程完成
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
