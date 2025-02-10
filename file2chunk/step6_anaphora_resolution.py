"""
文本指代消解和摘要生成工具（单文件版）

本脚本用于对单个JSON文件进行指代消解和摘要生成处理

使用方法：
    python script_name.py input.json output.json [--generate-summary]
"""

import json
import os
from typing import List, Dict
from openai import OpenAI
from datetime import datetime
import argparse
from collections import defaultdict
import tiktoken

# 初始化OpenAI客户端
MODEL_NAME = "gpt-4o-mini"
# api_key = "sk-4FuXXN5mrWi3jadx3f14E40f0f114eF4BbBaF00c7a58D303"
# api_key = "sk-T3zT8hsTB7DQbrjI6f80Ef96F1F74580B9462369Da44Ca9b"
api_key = "sk-X5VFivc7MW6CReAC4fF26bBdA81544A280Da66Dc4e103aD4"

client = OpenAI(
    api_key=api_key,
    base_url='https://az.gptplus5.com/v1',
)

def get_context_from_previous_chunks(chunks: List[Dict], current_idx: int, max_context: int = 4) -> str:
    """获取前文上下文"""
    context = []
    start_idx = max(0, current_idx - max_context)
    
    if current_idx == 0:
        return ""
    
    for i in range(start_idx, current_idx):
        if 0 <= i < len(chunks) and 'content' in chunks[i]:
            context.append(chunks[i]["content"])
    
    return " ".join(context)

def resolve_anaphora(text: str, context: str = "") -> str:
    """执行指代消解"""
    prompt = f''' 
    You are a language model assistant. Your task is to enhance the given text by replacing ambiguous or 
    context-dependent words with more specific and clear alternatives, based on the context of Lotus company's financial reports.

    Here are some guidelines for replacements:
    - Replace pronouns like "we" with the appropriate entity, but only use entities that appear in the current context:
      ✓ CORRECT: If the text mentions "Lotus Technology's R&D department" earlier, then "we" → "Lotus Technology's R&D department"
      ✗ INCORRECT: Don't introduce new entities or products that aren't mentioned in the context
    
    - When replacing "it" or other pronouns:
      ✓ CORRECT: Only use specific names/entities that were previously mentioned in the same or immediately preceding paragraph
      ✓ CORRECT: If the specific reference is unclear, use a descriptive but general term (e.g., "the company", "this initiative")
      ✗ INCORRECT: Don't introduce specific product names or entities that aren't in the source text

    - For product references:
      ✓ CORRECT: Only use exact product names that appear in the text
      ✓ CORRECT: If the specific product name isn't clear, use general terms like "the vehicle", "this model"
      ✗ INCORRECT: Don't make assumptions about which specific product is being discussed

    Examples:
    Original: "We developed it with cutting-edge technology."
    ✓ CORRECT (if previously mentioned): "Lotus Technology's engineering team developed the Eletre SUV with cutting-edge technology."
    ✓ CORRECT (if product unclear): "Lotus Technology's engineering team developed this vehicle with cutting-edge technology."
    ✗ INCORRECT: "Lotus Technology's engineering team developed the Elise with cutting-edge technology."

    Quality check:
    - Before making any replacement, verify that the entity or term exists in the current context
    - If unsure about the specific reference, prefer using more general but accurate terms
    - Never introduce new information that isn't present in the source text

    IMPORTANT - READ CAREFULLY:
    === REFERENCE CONTEXT (Use only for understanding references) ===
    {context}

    === TARGET TEXT (This is the ONLY text you should improve) ===
    {text}
    
    INSTRUCTIONS:
    1. Use the reference context ONLY to understand what pronouns and references refer to
    2. Modify and output ONLY the target text
    3. DO NOT include any part of the reference context in your output
    4. DO NOT add any explanations or notes - output only the improved text


    Please note:
    1. Use only the reference context to understand the direction of pronouns and referents
    2. Only modify and output the target text without repeating the reference context
    3. Do not include any explanations or additional information, only output the improved text
    '''
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a highly accurate and context-aware text editor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"API调用错误: {str(e)}")

def process_file(input_path: str, output_path: str, generate_summary: bool = True):
    """处理单个文件的主函数"""
    log_path = os.path.join(os.path.dirname(input_path), 
                          f"{os.path.splitext(os.path.basename(input_path))[0]}_processed.log")

    def log(message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

    try:
        log(f"开始处理文件: {input_path}")
        
        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理元数据
        metadata = data[0] if (data and isinstance(data[0], dict) and 'start' in data[0]) else None
        chunks = data[1:] if metadata else data

        processed_chunks = []
        previous_chunk = {}
        
        # 按标题分组处理
        title_groups = defaultdict(list)
        for chunk in chunks:
            title = chunk.get('title', '')
            title_groups[title].append(chunk)

        # 处理每个标题组
        for title, group in title_groups.items():
            log(f"正在处理标题组: {title}")
            
            # 生成摘要（如果需要）
            title_summary = ""
            if generate_summary:
                contents = [c['content'] for c in group if c.get('content')]
                if contents:
                    title_summary = generate_group_summary(title, contents)

            # 处理每个chunk
            for idx, chunk in enumerate(group):
                try:
                    # 指代消解处理
                    if 'content' in chunk:
                        context = get_context_from_previous_chunks(group, idx)
                        chunk['content'] = resolve_anaphora(chunk['content'], context)
                    
                        
                  
                    # 更新摘要信息
                    chunk['title_summary'] = f"title: {title}\nsummary: {title_summary}" if title_summary else ""
                    if 'title' in chunk:
                        del chunk['title']
                    
                    processed_chunks.append(chunk)
                  
                except Exception as e:
                    log(f"处理chunk {chunk.get('id')} 失败: {str(e)}")
                    processed_chunks.append(chunk)  # 保留原始数据

        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            output_data = [metadata] + processed_chunks if metadata else processed_chunks
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        log(f"处理完成，结果已保存到: {output_path}")

    except Exception as e:
        log(f"处理失败: {str(e)}")
        raise

def generate_group_summary(title: str, contents: List[str]) -> str:
    """生成分组摘要"""
    try:
        # 中间摘要生成
        intermediate = []
        for content in contents:
            summary = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": f"Please generate a summary for the following：\n{content}"
                }],
                temperature=0.3
            ).choices[0].message.content
            intermediate.append(summary)

        # 最终摘要整合
        final = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "Please combine the following summary, do not directly splice but make a certain summary：\n{}".format(''.join([str(item).replace('\\', '\\\\') for item in intermediate]))
        }],
        temperature=0.3
    ).choices[0].message.content
        
        return final
    except Exception as e:
        return f"摘要生成失败: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='单文件文本处理工具')
    parser.add_argument('input', help='输入JSON文件路径')
    parser.add_argument('output', help='输出JSON文件路径')
    parser.add_argument('--generate-summary', action='store_true', 
                      help='是否生成摘要（默认启用）', default=True)
    
    args = parser.parse_args()
    
    # 输入验证
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    if not args.input.endswith('.json'):
        raise ValueError("输入文件必须是JSON格式")
        
    process_file(args.input, args.output, args.generate_summary)

if __name__ == "__main__":
    main()
