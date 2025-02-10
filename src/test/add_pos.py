import sqlite3
import json

# 设置数据库路径和JSON文件路径
db_path = 'reranker_data.db'  # 替换为你的SQLite数据库路径
json_file_path = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/75_testingset.json'  # 替换为你的JSON文件路径

# 连接到SQLite数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 遍历JSON中的每个问题
for entry in json_data:
    question = entry['question']
    content_list = entry['content']
    
    # 对应SQLite中的query字段查找问题记录
    cursor.execute("SELECT id, query FROM old_55 WHERE query = ?", (question,))
    rows = cursor.fetchall()

    # 如果没有匹配的记录，跳过
    if not rows:
        continue

    # 遍历每个匹配的记录（可能会有多个）
    for row in rows:
        chunk_id = row[0]
        query = row[1]

        # 对比每条content，看看它是否匹配
        for i, content in enumerate(content_list):
            # 查找与当前content匹配的记录
            cursor.execute("SELECT id, chunks FROM old_55 WHERE id = ? AND chunks = ?", (chunk_id, content))
            matching_row = cursor.fetchone()

            if matching_row:
                # 如果找到匹配的记录，则将该记录的pos标记为1
                cursor.execute("UPDATE old_55 SET pos = 1 WHERE id = ?", (chunk_id,))
                print(f"Updated pos for chunk_id {chunk_id} to 1 for content {i+1}")

# 提交更新并关闭连接
conn.commit()
conn.close()

print("Pos tagging completed successfully.")
