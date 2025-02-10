import json
import os
import base64
import argparse
from pathlib import Path
import requests
from openai import OpenAI

# 修改后的提示词
prompt = """
Analyze and extract the table information from the image. Convert the table content into clear, structured natural language descriptions.

If NO table is found or the content is unclear, respond ONLY with "*" (a single asterisk).

Requirements:
1. Identify and describe each row's content accurately
2. Maintain all numerical values with their original units
3. Preserve any special formatting or notations
4. Include column headers in the description when relevant
5. Handle any footnotes or special marks in the table
6. Combine information from both table contents and surrounding context

Output Format:

[Table Level]
- Table Title: Identify the semantic title of the table (from caption or surrounding context)
- Table Summary: Provide a concise overview of what the table represents in 2-3 sentences
- Context: Summarize relevant information appearing before or after the table in 1-2 sentences
- Special Notes: Mention any important footnotes, units, or special formatting

[Row Level]
For each row, create a natural language description that:
- Combines column headers with cell values in a readable sentence
- Preserves numerical relationships and comparisons
- Includes relevant context from surrounding text
- Maintains semantic relationships between columns
- Uses proper units and formatting from the table

Format each row as: "Row [number]: [natural language description]"

Important:
- Keep all exact values and units
- Maintain technical terminology as is
- Be concise but complete
- Preserve any relationships between columns
- Include any qualifying conditions or constraints
"""


def load_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_gpt4(client, base64_image, prompt, context_before="", context_after=""):
    """增强的图片分析函数，包含上下文信息"""
    try:
        # 构建包含上下文的完整提示词
        full_prompt = f"""{prompt}

Context Information:
Before the table: {context_before}
After the table: {context_after}

Please analyze the table with this context in mind."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500  # 增加token限制以容纳更详细的输出
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in GPT-4 analysis: {str(e)}")
        return None

def get_context(data, current_idx, item):
    """获取表格的上下文，优先使用caption和footnote
    
    Args:
        data: 完整的数据列表
        current_idx: 当前项目的索引
        item: 当前项目数据
    
    Returns:
        tuple: (context_before, context_after)
    """
    context_before = ""
    context_after = ""
    
    # 1. 首先尝试使用caption和footnote
    if "img_caption" in item and item["img_caption"]:
        context_before = " ".join(item["img_caption"]).strip()
    
    if "img_footnote" in item and item["img_footnote"]:
        context_after = " ".join(item["img_footnote"]).strip()
    
    # 2. 如果caption或footnote为空，则查找周围文本
    if not context_before:  # 如果没有caption，向前查找
        j = current_idx - 1
        while j >= 0 and len(context_before.split()) < 100:
            if data[j].get('type') == 'text':
                context_before = data[j].get('text', '') + ' ' + context_before
            j -= 1
    
    if not context_after:  # 如果没有footnote，向后查找
        j = current_idx + 1
        while j < len(data) and len(context_after.split()) < 100:
            if data[j].get('type') == 'text':
                context_after += ' ' + data[j].get('text', '')
            j += 1
    
    return context_before.strip(), context_after.strip()

def process_json_file(json_path, output_base_dir):
    # 获取输入文件的名称（不含扩展名）
    input_filename = os.path.basename(json_path)
    input_filename = os.path.splitext(input_filename)[0]
    if input_filename.endswith('_content_list'):
        input_filename = input_filename[:-13]
    
    # 在输出基础目录下创建同名目录，替换空格为下划线
    output_dir = os.path.join(output_base_dir, input_filename.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    output_path = os.path.join(output_dir, 'base.json')
    
    # 获取JSON文件所在目录（用于找到相对路径的图片）
    json_dir = os.path.dirname(json_path)
    
    # 读取JSON文件
    print(f"Loading JSON file from: {json_path}")
    data = load_json_file(json_path)
    
    # 初始化输出数据
    output = []
    
    # 添加固定的第一个元素
    date_str = input_filename[:8] if len(input_filename) >= 8 else ""
    if date_str.isdigit():
        date_published = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    else:
        date_published = ""
        
    output.append({
        "start": 0,
        "end": 10000,
        "date_published": date_published
    })
    
    # 实时写入初始数据
    save_json_file(output, output_path)
    print(f"Initial metadata written to: {output_path}")
    
    # 初始化OpenAI客户端
    print("Initializing OpenAI client...")
    client = OpenAI(
        api_key="sk-X5VFivc7MW6CReAC4fF26bBdA81544A280Da66Dc4e103aD4",
        base_url='https://az.gptplus5.com/v1',
    )
    
    # 初始化当前标题和页码
    current_title = ""
    current_page = None
    
    # 处理每个项目
    for i, item in enumerate(data):
        # 记录标题，但不创建chunk
        if item.get('text_level') == 1:
            current_title = item.get('text', '')
            current_page = item.get('page_idx')
            print(f"Processing title: {current_title} on page {current_page}")
            continue  # 跳过后续处理，不创建chunk
        
        # 处理非标题内容
        if "type" in item:
            if item["type"] == "text":
                # 保存非标题文本内容
                text_chunk = {
                    "id": len(output),
                    "content": item.get('text', ''),
                    "page_number": item.get('page_idx', None),
                    "title": current_title,
                    "type": "text"
                }
                output.append(text_chunk)
                # 实时保存更新
                save_json_file(output, output_path)
                print(f"Text chunk written to: {output_path}")
            
            # 保留图片处理逻辑，但移到else分支
            elif "img_path" in item:
                print(f"Analyzing image at: {item['img_path']}")
                
                # 获取上下文（确保在这里获取）
                context_before, context_after = get_context(data, i, item)
                
                # 处理图片路径
                img_path = os.path.join(os.path.dirname(json_path), item['img_path'])
                
                try:
                    # 转换图片为base64
                    base64_image = image_to_base64(img_path)
                    
                    # 分析图片
                    analysis_result = analyze_image_with_gpt4(
                        client, 
                        base64_image, 
                        prompt,
                        context_before,  # 使用获取的上下文
                        context_after    # 使用获取的上下文
                    )
                    
                    if analysis_result and analysis_result != "*":
                        table_chunk = {
                            "id": len(output),
                            "content": analysis_result,
                            "page_number": item.get('page_idx', None),
                            "title": current_title,
                            "type": "table",
                            "context_source": {
                                "before": "caption" if item.get("img_caption") else "surrounding_text",
                                "after": "footnote" if item.get("img_footnote") else "surrounding_text"
                            },
                            "context_before": context_before,
                            "context_after": context_after
                        }
                        output.append(table_chunk)
                        save_json_file(output, output_path)
                        print(f"Table chunk written to: {output_path}")
                        
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}")
                    continue
    
    print(f"Final output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process JSON file with image analysis')
    parser.add_argument('input_path', help='Path to the input JSON file')
    parser.add_argument('output_base_dir', help='Base directory for output')
    
    args = parser.parse_args()
    
    try:
        process_json_file(args.input_path, args.output_base_dir)
        print(f"Successfully processed {os.path.basename(args.input_path)}")
    except Exception as e:
        print(f"Error processing {args.input_path}: {str(e)}")

if __name__ == "__main__":
    main()
