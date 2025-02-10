import json

def preview_jsonl(file_path: str, num_records=20, max_length=30):
    """预览JSONL文件的前几条记录"""
    total_records = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 第一遍遍历统计总记录数
        for line in f:
            total_records += 1
        
        # 第二遍遍历显示前N条
        f.seek(0)  # 重置文件指针
        for i, line in enumerate(f):
            if i >= num_records:
                break
                
            try:
                data = json.loads(line.strip())
                truncate = lambda s: s[:max_length] + '...' if len(s) > max_length else s
                
                query = truncate(data.get('query', 'N/A'))
                pos = [truncate(p) for p in data.get('pos', [])]
                neg = [truncate(n) for n in data.get('neg', [])]
                prompt = truncate(data.get('prompt', 'N/A'))
                
                print(f"Record {i+1}:")
                print(f"Query: {query}")
                print(f"Pos: {pos}")
                print(f"Neg: {[n for n in neg[:2]]}{'...' if len(neg)>2 else ''} ({len(neg)} items)")
                print(f"Prompt: {prompt}")
                print("-" * 50)
                
            except json.JSONDecodeError:
                print(f"Error parsing line {i+1}")
                print("-" * 50)
    
    # 最后打印总记录数
    print(f"\n\033[1mTotal records in file: {total_records}\033[0m")

if __name__ == "__main__":
    # jsonl_file = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/reranker_data_0120/origin/reranker_data.jsonl"
    jsonl_file = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/reranker_data.jsonl"
    preview_jsonl(jsonl_file)