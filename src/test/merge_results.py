import os

def merge_jsonl_files(input_dir, output_file):
    """
    合并指定目录下所有 JSONL 文件的记录到一个新的 JSONL 文件中。

    Args:
        input_dir (str): 输入目录路径，包含要合并的 JSONL 文件。
        output_file (str): 输出的合并后的 JSONL 文件路径。
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        files_processed = 0
        records_written = 0

        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                line = line.strip()
                                if line:  # 确保行不为空
                                    outfile.write(line + '\n')
                                    records_written += 1
                        files_processed += 1
                        print(f"已处理文件: {file_path}, 写入记录数: {records_written}")
                    except Exception as e:
                        print(f"读取文件 {file_path} 时出错: {e}")

    print(f"合并完成。处理了 {files_processed} 个文件，总共写入了 {records_written} 条记录到 '{output_file}'")

# 示例用法
if __name__ == "__main__":
    # 指定输入目录和输出文件
    input_dir = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/reranker_data_20250109/origin'
    output_file = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/reranker_data_20250109/merged/merged.jsonl'

    # 执行合并操作
    merge_jsonl_files(input_dir, output_file)
