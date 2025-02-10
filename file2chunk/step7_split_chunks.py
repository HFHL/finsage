"""
文本内容分块处理工具

本脚本用于对 JSON 格式的文本内容进行分块处理，确保每块的大小符合指定的单词数量限制，同时保持句子的完整性。主要功能包括：
- 根据句子边界分割长文本为指定大小的块。
- 合并相同标题的文本内容，并规范化页码范围。
- 跳过元数据和表格类型内容，保留它们的完整性。
- 输出为新的 JSON 文件。

功能特点：
1. **文本分块**：
   - 按指定的单词数量限制将文本内容分块，确保每个块的句子完整。
   - 支持智能分割，避免拆分句子。

2. **页码规范化**：
   - 合并连续或重复的页码，生成范围表示。

3. **分块规则**：
   - 相同标题下的内容自动合并。
   - 当文本块过大时，按单词数限制进行分割。

主要模块和方法：
1. `split_content(content: str, content_size: int) -> List[str]`：
   - 按指定大小分割文本内容，确保句子完整性。
   - 使用正则表达式匹配句子边界，同时保护特殊情况（如缩写和小数点）。

2. `normalize_page_range(page_str: str) -> str`：
   - 简化页码处理，只返回第一个页码。

3. `process_json(input_file: str, output_file: str, content_size: int = 200)`：
   - 处理输入 JSON 文件：
     - 跳过元数据（metadata）和表格类型内容。
     - 合并同一标题下的内容，并按大小分块。
     - 将结果写入新的 JSON 文件。

4. `main()`：
   - 提供命令行接口，支持用户指定输入文件、输出文件和分块大小。

命令行使用说明：
    python script_name.py -i input.json -o output.json [-s 块大小]


参数：
- `-i`, `--input`：输入的 JSON 文件路径（必填）。
- `-o`, `--output`：输出的 JSON 文件路径（必填）。
- `-s`, `--size`：每块的单词数限制，默认值为 200。

输出：
    处理后的文件将保存为新文件，文件名格式为命令行所填写的output.json

"""

import json
import re
import argparse

def split_content(content, content_size):
    """将文本内容分割成指定大小的块，确保句子完整性"""
    # 定义句子分隔模式 - 修改正则表达式以更准确地识别句子边界
    sentence_pattern = r'(?<=[.。!！?？；;])\s+'
    
    # 预处理：保护特殊情况
    protected_text = content
    # 保护常见缩写
    abbreviations = ['Mr.', 'Mrs.', 'Dr.', 'Ph.D.', 'etc.', 'i.e.', 'e.g.', 'U.S.', 'Inc.', 'Ltd.', 'Co.', 'U.K.']  # 添加 U.S.
    for abbr in abbreviations:
        protected_text = protected_text.replace(abbr, abbr.replace('.', '@POINT@'))
    
    # 保护数字中的小数点
    protected_text = re.sub(r'(\d+)\.(\d+)', r'\1@POINT@\2', protected_text)
    
    # 分割句子
    sentences = re.split(sentence_pattern, protected_text)
    sentences = [s.strip().replace('@POINT@', '.') for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # 如果当前句子加上已有内容不超过限制，添加到当前块
        if current_word_count + sentence_words <= content_size:
            current_chunk.append(sentence)
            current_word_count += sentence_words
        else:
            # 如果当前块不为空，保存它
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            # 开始新的块，从当前句子开始
            current_chunk = [sentence]
            current_word_count = sentence_words
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def normalize_page_range(page_str):
    """简化页码处理，只返回第一个页码
    
    输入: "112-112-113-114-114-115"
    输出: "112"
    """
    # 分割页码并返回第一个
    pages = page_str.split('-')
    return pages[0]

def process_json(input_file, output_file, content_size=200):
    print(f"Starting to process {input_file}")
    chunks = []
    chunk_id_counter = 1  # Initialize a counter for unique IDs

    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        items = json.load(f)

    index = 0
    while index < len(items):
        item = items[index]
        # 确保必要的字段存在
        if 'type' not in item or 'content' not in item:
            print(f"Skipping invalid item: {item}")
            index += 1
            continue

        # 为没有title_summary的项目添加空title_summary
        if 'title_summary' not in item:
            item['title_summary'] = "title: \nsummary: "

        if item['type'] == 'table':
            # 表格类型直接保存
            item['id'] = chunk_id_counter
            chunk_id_counter += 1
            chunks.append(item)
            index += 1
            continue

        current_chunk = item.copy()
        current_title = current_chunk.get('title_summary', '')
        content_words = len(current_chunk['content'].split())

        # 如果内容过短，尝试从后续块中拼接
        while content_words < content_size and index + 1 < len(items):
            next_item = items[index + 1]
            next_title = next_item.get('title_summary', '')
            # Only merge if the titles are the same
            if next_item['type'] == 'text' and next_title == current_title:
                # Extract content from the next item
                additional_content = next_item['content']
                current_chunk['content'] += ' ' + additional_content
                # Keep the original page_number
                current_chunk['page_number'] = current_chunk['page_number']
                # Remove the next item since its content has been merged
                items.pop(index + 1)
                content_words = len(current_chunk['content'].split())
            else:
                break

        # 如果内容过长，拆分最后的句子
        if content_words > content_size:
            sentences = split_content(current_chunk['content'], content_size)
            # 当前chunk取分割后的第一部分
            current_chunk['content'] = sentences[0]
            current_chunk['id'] = chunk_id_counter
            chunk_id_counter += 1
            chunks.append(current_chunk)
            # 剩余的句子作为新的chunk
            remaining_content = ' '.join(sentences[1:])
            if remaining_content.strip():
                # 插入到items列表中，以便下次循环处理
                new_item = item.copy()
                new_item['content'] = remaining_content
                items.insert(index + 1, new_item)
        else:
            current_chunk['id'] = chunk_id_counter
            chunk_id_counter += 1
            chunks.append(current_chunk)
        index += 1

    print(f"Writing {len(chunks)} chunks to {output_file}")
    # 写入为JSON格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Written {len(chunks)} chunks.")

def main():
    parser = argparse.ArgumentParser(description='文本分块处理工具')
    parser.add_argument('-i', '--input', required=True, help='输入的JSON文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出的JSON文件路径')
    parser.add_argument('-s', '--size', type=int, default=200, help='每块的最小单词数（默认：200）')
    
    args = parser.parse_args()
    
    process_json(args.input, args.output, args.size)  # 改用process_json
    print(f'处理完成！\n输入文件：{args.input}\n输出文件：{args.output}\n块大小：{args.size}')

if __name__ == '__main__':
    main()
