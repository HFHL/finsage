import sqlite3
import json
import os

def fetch_data_from_db(db_path):
    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取所有数据
    cursor.execute("SELECT query, answer, chunks, pos, rewriten_q FROM old_55")
    rows = cursor.fetchall()
    
    conn.close()
    return rows

def process_data(rows, output_file):
    query_neg_map = {}  # 用于存储每个问题的负例列表
    all_data = []  # 用于存储所有的 JSONL 数据
    processed_queries = set()  # 用于存储已处理过的正例的查询，避免重复添加

    # 第一步：收集所有的负例（pos != 1）
    for row in rows:
        query, answer, chunks, pos, rewriten_q = row
        
        # 确保chunks不是空的
        if not chunks:
            continue

        # 只从rewriten_q获取问题
        query = rewriten_q
        
        # 处理负例 (如果 pos 不是 1 的记录)
        if pos != 1:
            if query not in query_neg_map:
                query_neg_map[query] = []
            query_neg_map[query].append(chunks)

    # 第二步：遍历数据，为每个正例生成 JSONL 记录
    for row in rows:
        query, answer, chunks, pos, rewriten_q = row
        
        # 确保chunks不是空的
        if not chunks:
            continue

        # 只从rewriten_q获取问题
        query = rewriten_q
        
        # 处理正例 (只处理 pos 为 1 的记录)
        if pos == 1:
            # 获取该问题的负例，避免重复
            neg_examples = query_neg_map.get(query, [])
            # 排除正例的chunks，只保留其他负例
            neg_examples = [neg for neg in neg_examples if neg != chunks]

            # 创建正例 JSONL 数据
            data = {
                "query": query,
                "pos": [chunks] if not isinstance(chunks, list) else chunks,  # 确保 pos 是列表
                "neg": neg_examples if neg_examples else ["No other negative examples found."]
            }
            all_data.append(data)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 写入所有数据到 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_data:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

def main(db_path, output_file):
    # 1. 从数据库中获取数据
    rows = fetch_data_from_db(db_path)

    # 2. 处理数据并生成 JSONL 文件
    process_data(rows, output_file)

if __name__ == '__main__':
    # 指定数据库路径和输出文件路径
    db_path = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/reranker_data.db'  # 数据库路径
    output_file = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/reranker_data_0202/origin/reranker_0202.jsonl'  # 输出文件路径

    # 执行程序
    main(db_path, output_file)
