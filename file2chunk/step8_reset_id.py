import json

def reset_ids(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 重设 id，从 1 开始
    current_id = 1
    for item in data:
        if 'id' in item:  # 只处理有 id 字段的项
            item['id'] = current_id
            current_id += 1
        # item['id'] = current_id
        # current_id += 1
        
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 使用示例
file_path = '/root/autodl-tmp/file2chunk/script/1121ppt.json'
reset_ids(file_path)