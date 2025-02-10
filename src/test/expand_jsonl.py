import json
import argparse
from tqdm import tqdm

def expand_jsonl(input_path, output_path, multiplier=10):
    """
    Read JSONL file, expand each record, and write to new JSONL file
    """
    # Read input file and expand records
    expanded_records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
        
    print(f"Found {len(records)} original records")
    
    # Expand each record
    for record in tqdm(records, desc="Expanding records"):
        for i in range(multiplier):
            # Create a copy of the record with a unique identifier
            new_record = record.copy()
            new_record['copy_id'] = i + 1
            expanded_records.append(new_record)
    
    # Write expanded records to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in expanded_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Written {len(expanded_records)} expanded records to {output_path}")

def main():
    input_path = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/reranker_data_0202/origin/reranker_0202.jsonl"  # Replace with your input file path
    output_path = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/reranker_data_0202/expand/reranker_data_0202.jsonl"  # Replace with your output file path
    multiplier = 6  # Set your desired multiplier
    
    
    
    expand_jsonl(input_path, output_path, multiplier)

if __name__ == '__main__':
    main()