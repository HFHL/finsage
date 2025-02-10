"""
JSON文件标题字段删除工具

这个脚本用于处理包含文档chunks的JSON文件，主要功能是删除每个chunk中的'title'字段。

预期的JSON文件结构：
[
    {metadata...},  # 第一个元素为元数据（保持不变）
    {
        "id": int,      # chunk的ID
        "content": str, # chunk的内容
        "title": str,   # 将被删除的标题字段
        ...
    },
    ...
]

使用说明：
1. 脚本会自动创建新的输出文件，文件名后缀为'_no_title'
2. 原始文件不会被修改
3. 处理完成后会显示处理的数据条数和输出文件路径

使用示例：

1. 命令行方式：
    python step3(optioinal)_remove_title.py ./data/base.json
    # 将在同目录创建 ./data/base_no_title.json

2. 作为模块导入：
    from step3_remove_title import remove_title_field
    
    # 处理单个文件
    remove_title_field("./data/base.json")
    # 输出: ./data/base_no_title.json
    
    # 批量处理文件
    import glob
    for json_file in glob.glob("./data/*.json"):
        remove_title_field(json_file)

输入文件示例：
[
    {"version": "1.0"},
    {
        "id": 1,
        "content": "示例内容",
        "title": "将被删除的标题"
    }
]

输出文件示例：
[
    {"version": "1.0"},
    {
        "id": 1,
        "content": "示例内容"
    }
]

作者：Hailin He
版本：1.0.0
日期：2024-11-18
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def remove_title_field(file_path: str) -> None:
    """
    Remove the 'title' field from each item in the JSON file.
    
    Args:
        file_path (str): Path to the JSON file
    """
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Remove 'title' field from each item (except the first metadata item)
        for item in data[1:]:  # Skip the first item (metadata)
            if 'title' in item:
                del item['title']
        
        # Create output path with '_no_title' suffix
        output_path = str(Path(file_path).with_stem(Path(file_path).stem + '_no_title'))
        
        # Write the modified data back to a new file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully processed {len(data)-1} items")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python step3(optioinal)_remove_title.py <json_file_path>")
        print("Example: python step3(optioinal)_remove_title.py ./base.json")
        sys.exit(1)
        
    json_file_path = sys.argv[1]
    remove_title_field(json_file_path)

# 删除或注释掉原来的示例调用
# remove_title_field("./base.json")