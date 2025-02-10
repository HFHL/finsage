"""
基于相似度分析的文本去重工具

本脚本用于根据相似度分析结果，从基础 JSON 文件中删除重复的文本块（chunks）。
它读取包含相似度对的 JSON 文件，提取重复的 chunk ID，并在基础文件中删除这些重复项。

功能特点：
- 从相似度分析文件中提取重复的 chunk ID。
- 根据重复的 chunk ID，过滤基础文件中的重复项。
- 输出去重后的新 JSON 文件，保留原文件的完整性。

模块和方法：
1. `get_duplicate_ids(similarity_file: str) -> Set[int]`：
   - 从相似度分析文件中提取重复的 chunk ID。
   - 读取 `similar_pairs` 字段，收集所有 chunk2 的 ID，作为需要删除的重复项。
   - 返回重复的 chunk ID 集合。

2. `process_base_file(base_file: str, duplicate_ids: Set[int]) -> None`：
   - 读取基础文件，删除重复的 chunks。
   - 过滤掉 ID 存在于 `duplicate_ids` 中的 chunks。
   - 将去重后的数据保存到一个新的 JSON 文件。


命令行使用说明：
python step5_delete_similar_chunks.py similarity_file base_file


参数：
- `similarity_file`: 包含相似度分析结果的 JSON 文件路径。
- `base_file`: 待去重的基础 JSON 文件路径。

输出：
- 生成一个新的 JSON 文件，文件名格式为 `{base_file}_deduped.json`，存储去重后的数据。

输入文件格式要求：
**相似度分析文件**（示例）：
   ```json
   {
       "similar_pairs": [
           {
               "chunk1": {"id": 1, "content": "文本内容1"},
               "chunk2": {"id": 2, "content": "文本内容2"},
               "similarity": 0.95
           },
           ...
       ]
   }

"""


import json
import argparse
import os
from typing import List, Set

def get_duplicate_ids(similarity_file: str) -> Set[int]:
    """从相似度文件中提取需要删除的chunk id"""
    print(f"Reading similarity file: {similarity_file}")
    
    with open(similarity_file, 'r', encoding='utf-8') as f:
        sim_data = json.load(f)
    
    duplicate_ids = set()
    pair_count = len(sim_data.get('similar_pairs', []))
    print(f"Found {pair_count} similar pairs")
    
    for pair in sim_data.get('similar_pairs', []):
        # 保留chunk1的id,收集其他所有chunk的id
        chunk1_id = pair['chunk1']['id']
        chunk2_id = pair['chunk2']['id']
        similarity = pair['similarity']
        
        # 只添加chunk2的id到待删除集合
        duplicate_ids.add(chunk2_id)
        print(f"Found duplicate: chunk1_id={chunk1_id}, chunk2_id={chunk2_id}, similarity={similarity:.4f}")
    
    print(f"\nTotal {len(duplicate_ids)} duplicate IDs identified")
    return duplicate_ids

def process_base_file(base_file: str, duplicate_ids: Set[int]) -> None:
    """处理基础文件,删除重复的chunks"""
    print(f"\nProcessing base file: {base_file}")
    
    with open(base_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    original_count = len(base_data)
    print(f"Original chunk count: {original_count}")
    
    # 检查第一个元素的结构并打印
    if original_count > 0:
        print(f"First chunk structure: {list(base_data[0].keys())}")
    
    # 过滤掉重复的chunks，添加错误处理
    filtered_data = []
    for chunk in base_data:
        try:
            if chunk.get('id') not in duplicate_ids:
                filtered_data.append(chunk)
        except (KeyError, TypeError) as e:
            print(f"Warning: Invalid chunk structure: {chunk}")
            continue
    
    # 生成新文件名
    file_name, file_ext = os.path.splitext(base_file)
    output_file = f"{file_name}_deduped{file_ext}"
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    final_count = len(filtered_data)
    removed_count = original_count - final_count
    print(f"Removed {removed_count} duplicate chunks")
    print(f"Final chunk count: {final_count}")
    print(f"Output saved to: {output_file}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Remove duplicate chunks based on similarity analysis')
    parser.add_argument('similarity_file', help='Path to the similarity analysis JSON file')
    parser.add_argument('base_file', help='Path to the base JSON file')
    
    args = parser.parse_args()
    
    # 执行处理流程
    print("=== Starting Deduplication Process ===\n")
    
    try:
        # 获取重复ID
        duplicate_ids = get_duplicate_ids(args.similarity_file)
        
        # 处理基础文件
        process_base_file(args.base_file, duplicate_ids)
        
        print("\n=== Deduplication Process Completed Successfully ===")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()