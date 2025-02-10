import sqlite3
import json

def create_table_if_not_exists(conn):
    # 创建表格（如果不存在的话）
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS old_55 (
            id INTEGER PRIMARY KEY AUTOINCREMENT, -- 自动增长的ID
            query TEXT NOT NULL, -- 问题，字符串类型
            answer TEXT NOT NULL, -- 回答，字符串类型
            chunks TEXT NOT NULL, -- 分段数据，存储为JSON字符串类型（List[str]）
            pos INTEGER, -- 布尔值，表示是否为正例，允许为NULL
            rewriten_q TEXT -- 重写后的问题，字符串类型
        )
    ''')
    conn.commit()

def parse_json_to_db(json_file, db_path):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    
    # 创建表格（如果不存在）
    create_table_if_not_exists(conn)
    
    cursor = conn.cursor()

    # 打开并读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # 遍历JSON中的每个记录
        for record in data:
            question = record.get("question")
            rewritten_question = record.get("rewritten")
            answer = record.get("answer")
            chunks = record.get("all_retrieved", [])
            
            # 遍历all_retrieved中的每一项，作为单独的记录插入
            for chunk in chunks:
                if chunk:  # 确保chunk不为空
                    cursor.execute('''
                        INSERT INTO old_55 (query, answer, chunks, rewriten_q) 
                        VALUES (?, ?, ?, ?)
                    ''', (question, answer, chunk, rewritten_question))

    # 提交并关闭数据库连接
    conn.commit()
    conn.close()

# 使用方法
json_file = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/scenarios/5/hyde_reranker.json'  # 替换为你的JSON文件路径
db_path = 'reranker_data.db'  # 替换为你的SQLite数据库路径

parse_json_to_db(json_file, db_path)
