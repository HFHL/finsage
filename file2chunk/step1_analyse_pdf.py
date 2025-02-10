"""
使用方法：
    python step1_analyse_pdf.py --pdf_path pdf的位置 --out_dir output的文件夹
"""

import argparse
from pathlib import Path
import os
import json
import copy
from bs4 import BeautifulSoup
from typing import List, Dict
import re
from dataclasses import dataclass
import requests
from loguru import logger
from openai import OpenAI
import base64
import time

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
import magic_pdf.model as model_config
import logging
from tqdm import tqdm

model_config.__use_inside_model__ = True

BASE_URL = "http://127.0.0.1:11434/v1/"
MODEL_NAME = "qwen2:72b"


def json_md_dump(
        pipe,
        md_writer,
        pdf_name,
        content_list,
        md_content=None,
):
    # 写入模型结果到 model.json
    orig_model_list = copy.deepcopy(pipe.model_list)
    md_writer.write(
        content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_model.json"
    )

    # 写入中间结果到 middle.json
    md_writer.write(
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_middle.json"
    )

    # text文本结果写入到 conent_list.json
    md_writer.write(
        content=json.dumps(content_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_content_list.json"
    )

    if md_content is not None:
        # 写入结果到 .md 文件中
        md_writer.write(
            content=md_content,
            path=f"{pdf_name}.md"
        )


@dataclass
class HeaderCell:
    text: str
    start_col: int
    end_col: int
    level: int


def extract_header_structure(thead: BeautifulSoup) -> List[HeaderCell]:
    """
    Extract header cells with their column spans and levels.

    Args:
        thead (BeautifulSoup): Table head section

    Returns:
        List[HeaderCell]: List of header cells with position information
    """
    header_cells = []
    header_rows = thead.find_all('tr')

    for level, row in enumerate(header_rows):
        current_col = 0
        cells = row.find_all(['td', 'th'])

        for cell in cells:
            # Clean text
            text = cell.get_text(strip=True)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'</?b>', '', text)

            # Get span attributes
            colspan = int(cell.get('colspan', 1))

            if text:  # Only process non-empty cells
                end_col = current_col + colspan - 1
                header_cells.append(HeaderCell(
                    text=text,
                    start_col=current_col,
                    end_col=end_col,
                    level=level
                ))

            current_col += colspan

    return header_cells


def get_column_headers(header_cells: List[HeaderCell], column_index: int) -> List[str]:
    """
    Get all headers that apply to a specific column.

    Args:
        header_cells (List[HeaderCell]): List of all header cells
        column_index (int): The column index to find headers for

    Returns:
        List[str]: List of headers that apply to this column
    """
    applicable_headers = []
    for cell in header_cells:
        if cell.start_col <= column_index <= cell.end_col:
            applicable_headers.append((cell.level, cell.text))

    # Sort by level and extract just the text
    return [h[1] for h in sorted(applicable_headers)]


def process_html_table(html_content: str, page_num: int, chunk_size: int = 3) -> List[Dict]:
    """
    Process HTML table content with properly spanned multi-level headers.

    Args:
        html_content (str): HTML table string
        chunk_size (int): Number of rows per chunk

    Returns:
        List[Dict]: List of chunks with processed table content
    """
    try:
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the innermost table if the table is wrapped in td
        table = soup.find('table')
        if not table:
            return []

        # Extract header structure
        thead = table.find('thead')
        if not thead:
            return []
        
        header_cells = extract_header_structure(thead)

        # Process rows
        rows = []
        tbody = table.find('tbody')
        if not tbody:
            return []

        for row in tbody.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue

            # Get the description (first column)
            description = cells[0].get_text(strip=True)
            description = re.sub(r'\s+', ' ', description)

            # Process data cells (excluding first column)
            data_parts = []
            for col_idx, cell in enumerate(cells[1:], start=1):  # Start from 1 to skip description
                cell_text = cell.get_text(strip=True)
                if cell_text:
                    # Get all headers that apply to this column
                    column_headers = get_column_headers(header_cells, col_idx)
                    if column_headers:
                        header_text = ' - '.join(column_headers)
                        data_parts.append(f"{header_text}: {cell_text}")

            # Format the complete row
            if description:
                row_text = f"{description}: {', '.join(data_parts)}"
            else:
                row_text = ', '.join(data_parts)

            rows.append(row_text)

    except Exception as e:
        logger.exception(e)

    # Create chunks
    chunks = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]

        # Join rows with newlines
        chunk_text = '\n'.join(chunk)

        # Create chunk metadata
        chunk_dict = {
            'content': chunk_text,
            'row_count': len(chunk),
            'start_row': i,
            'page_number': page_num
            # 'header_structure': [(h.text, h.start_col, h.end_col, h.level)
            #                    for h in header_cells]
        }
        chunks.append(chunk_dict)

    return chunks


def table_sys_prompt():
    return f'''
Help analyze some specific tables that I'll provide. You are tasked with extracting and structuring information from a table and its surrounding context. You should extract both table-level and row-level information for the table.

Output format:

[Tabe Level]
Extract and structure the following:
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

Format each row description as: "Row {{index}}: {{natural language description}}"

Example Output:
[Table Level]
Title: Global Renewable Energy Adoption Rates in 2023 first half
Summary: This table compares renewable energy adoption across countries, showing installation capacity and growth rates from 2020-2023.
Context: The report examines renewable energy growth across major economies. These adoption rates suggest an accelerating transition to clean energy.
Notes: Capacity values are in gigawatts (GW). Growth rates adjusted for seasonal variations. 

[Row Level]
Row 1: China leads renewable energy adoption with 573 GW installed capacity and 12.3% year-over-year growth, having completed most planned installations (A).
Row 2: The United States follows with 325 GW capacity, showing 8.7% growth while several projects remain under development.

Rules:
1. Preserve all numerical values and units
2. Maintain relationships between columns
3. Include contextual information when relevant
4. Use consistent formatting
5. Keep descriptions concise but informative

Remember to combine the information from both table contents and surrounding context to create comprehensive and accurate descriptions.
'''


def table_user_prompt(table, ctx_before, ctx_after) -> str:
    ret = f'''
Help analyze the following table given the context provided.

[Context before Table]
{ctx_before}

[Context after Table]
{ctx_after}

'''
    if table is not None:
        ret += f'[Table Content]\n{table}'
    return ret


qwen_client = OpenAI(
    base_url=BASE_URL,
    api_key="ollama"
)

# kimi API KEY own by WYX
api_key = "sk-vtqu733JfyRnqXj8XGk28I2VPlnGKpEZ9wurIZvOIm1J9jye"
kimi_client = OpenAI(
    api_key=api_key,
    base_url='https://api.moonshot.cn/v1',
)

# openai API KEY own by CJJ
gpt_key = "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"
gpt_client = OpenAI(
    api_key=gpt_key
)


def process_table_llm(ctx_before: str, ctx_after: str, page_num: int, chunk_size: int, llm: str,
                      table_content: str = None, table_img_path: str = None) -> List[Dict]:
    messages = [
        {"role": "system", "content": table_sys_prompt()},
        {"role": "user", "content": table_user_prompt(table_content, ctx_before, ctx_after)}
    ]

    try:
        # qwen
        # data = {
        #     "model": MODEL_NAME,
        #     "messages": messages,
        #     "stream": False
        # }
        # response = requests.post(f"{BASE_URL}/api/chat", json=data, stream=False)
        # response.raise_for_status()
        # assert response.status_code == 200
        # response_data = response.json()['message']['content']

        if llm == "kimi":
            # parse table page from image
            if table_img_path is not None:
                file_obj = kimi_client.files.create(file=Path(table_img_path), purpose="file-extract")
                file_content = kimi_client.files.content(file_id=file_obj.id).text
                messages[1]["content"] = table_user_prompt(file_content, ctx_before, ctx_after)

            completion = kimi_client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=messages,
                temperature=0.3,
            )

            # completion = qwen_client.chat.completions.create(
            #     model="qwen2:72b",
            #     messages=messages,
            #     temperature=0.3,
            # )
        elif llm == "gpt":
            # use gpt-4o for image extraction (need proxy)
            if table_img_path is None:
                raise ValueError("table_img_path is required for gpt-4o")

            with open(table_img_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
            completion = gpt_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", "content": table_sys_prompt()
                    },
                    {
                        "role": "user", "content": [
                        {"type": "text", "text": table_user_prompt(table_content, ctx_before, ctx_after)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                    }
                ]
            )
        else:
            raise ValueError(f"Invalid LLM model: {llm}")

        response_data = completion.choices[0].message.content

        chunks = []
        table_info, row_info = response_data.split("[Row Level]")
        table_info = table_info.split("[Table Level]")[1].strip()

        rows = row_info.strip().split("\n")

        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            chunk_text = table_info + '\n' + '\n'.join(chunk)
            # Create chunk metadata
            chunk_dict = {
                'content': chunk_text,
                'row_count': len(chunk),
                'start_row': i,
                'page_number': page_num
            }
            chunks.append(chunk_dict)

        return chunks

    except Exception as e:
        logger.exception(e)
        return []


def parse_table(
        pdf_path: str,
        start_page_id,
        end_page_id,
        out_dir: str,
        extract_table_text: bool,
        chunk_row_size: int = 30,
        parse_method: str = 'auto',
        model_json_path: str = None,
        is_json_md_dump: bool = True,
):
    """
    执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到 pdf 文件所在的目录

    :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
    :param parse_method: 解析方法， 共 auto、ocr、txt 三种，默认 auto，如果效果不好，可以尝试 ocr
    :param model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
    :param is_json_md_dump: 是否将解析后的数据写入到 .json 和 .md 文件中，默认 True，会将不同阶段的数据写入到不同的 .json 文件中（共3个.json文件），md内容会保存到 .md 文件中
    :param out_dir: 输出结果的目录地址，保存所有结果
    :param extract_table_text: 是否提取表格文本
    """
    try:
        pdf_name = os.path.basename(pdf_path).split(".")[0]

        output_image_path = os.path.join(out_dir, 'images')

        # 获取图片的父路径，为的是以相对路径保存到 .md 和 conent_list.json 文件中
        image_path_parent = os.path.basename(output_image_path)

        pdf_bytes = open(pdf_path, "rb").read()  # 读取 pdf 文件的二进制数据

        if model_json_path:
            # 读取已经被模型解析后的pdf文件的 json 原始数据，list 类型
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # 执行解析步骤
        # image_writer = DiskReaderWriter(output_image_path)
        image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(out_dir)

        # 选择解析方式
        # jso_useful_key = {"_pdf_type": "", "model_list": model_json}
        # pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        if parse_method == "auto":
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer, False, start_page_id=start_page_id,
                           end_page_id=end_page_id, formula_enable=False, table_enable=False)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            logger.error("unknown parse method, only auto, ocr, txt allowed")
            exit(1)

        # 执行分类
        pipe.pipe_classify()

        # 如果没有传入模型数据，则使用内置模型解析
        if not model_json:
            if model_config.__use_inside_model__:
                pipe.pipe_analyze()  # 解析
            else:
                logger.error("need model list input")
                exit(1)

        # 执行解析
        pipe.pipe_parse()

        # 保存 text 和 md 格式的结果
        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        # md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

        if is_json_md_dump:
            json_md_dump(pipe, md_writer, pdf_name, content_list)

        chunk_list = []
        for i, content in enumerate(content_list):
            if content['type'] != 'table':
                continue

            ctx_before = "\n".join(content['table_caption'])
            ctx_after = "\n".join(content['table_footnote'])
            j = i - 1
            cnt = 0
            while j >= 0 and cnt < 2 and content_list[j]['type'] == "text":
                ctx_before = content_list[j]['text'] + '\n' + ctx_before
                cnt += 1
                j -= 1

            j = i + 1
            cnt = 0
            while j < len(content_list) and cnt < 2 and content_list[j]['type'] == "text":
                ctx_after = ctx_after + '\n' + content_list[j]['text']
                cnt += 1
                j += 1

            if extract_table_text:
                while True:
                    try:
                        table_html = content['table_body'].strip()
                        table_img_path = os.path.join(out_dir, content['img_path'])
                        chunks = process_table_llm(ctx_before, ctx_after, llm='kimi', page_num=content['page_idx'] + 1,
                                                   chunk_size=chunk_row_size, table_content=table_html)
                        break
                    except Exception as e:
                        error_message = str(e)
                        if "rate_limit_reached_error" in error_message:
                            print("Rate limit reached. Sleeping for 15 seconds...")
                            time.sleep(15)
                        elif "high risk" in error_message:
                            print("High risk error detected. Sleeping for 1 minute...")
                            time.sleep(60)
                        else:
                            print(f"Error: {error_message}")
                            break

                if len(chunks):
                    chunk_list.append(*chunks)

                for i, chunk in enumerate(chunks):
                    print(f"\n\nChunk {i + 1}:")
                    print(chunk)
            else:
                chunk_list.append(
                    {
                        'ctx_before': ctx_before,
                        'ctx_after': ctx_after,
                        'page_num': content['page_idx'] + 1,
                        'img_path': content['img_path'],
                    }
                )
        return chunk_list


    except Exception as e:
        logger.exception(e)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Parse tables from PDF file')

    parser.add_argument(
        '--pdf_path',
        type=str,
        required=True,
        help='Path to the PDF file',
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        help='Path to save the output JSON file. If not provided, will use PDF name with _table.json suffix'
    )

    parser.add_argument(
        '--start_page',
        type=int,
        default=1,
        help='Start page number (default: 1)'
    )

    parser.add_argument(
        '--end_page',
        type=int,
        default=10000,
        help='End page number (default: 10000)'
    )

    parser.add_argument(
        '--chunk_row_size',
        type=int,
        default=30,
        help='Number of rows per chunk (default: 30)'
    )

    parser.add_argument(
        '--extract_table_text', '-ext',
        type=bool,
        default=False,
        help='Extract table text using LLM (default: True)'
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # pdf_path = "/Users/yuchenhua/Downloads/Lotus F-1 20240919.pdf"
    # out_dir = "/Users/yuchenhua/Downloads"

    # Parse tables from PDF
    table_chunks = parse_table(
        args.pdf_path,
        is_json_md_dump=True,
        start_page_id=args.start_page - 1,
        end_page_id=args.end_page - 1,
        # out_dir=out_dir,
        out_dir=args.out_dir,
        chunk_row_size=args.chunk_row_size,
        extract_table_text=args.extract_table_text
    )

    # Extract and format date from filename
    date_match = re.search(r'(\d{8})', args.pdf_path)
    # date_match = re.search(r'(\d{8})', pdf_path)
    if date_match:
        date_str = date_match.group(1)
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        pdf_metadata = [{
            "start": 0,
            "end": 10000,
            "date_published": formatted_date
        }]
        all_data = pdf_metadata + table_chunks
    else:
        print('Missing date_pulished. Please add it before storing in vector db.')
        all_data = table_chunks

    # Save results to JSON file
    output_path = os.path.join(args.out_dir, Path(args.pdf_path).stem + '_table.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks to: {output_path}")
    input_json = output_path
    img_dir = os.path.join(args.out_dir, 'images')
    output_json = os.path.join(args.out_dir, Path(args.pdf_path).stem + '_table_chunk.json')
    process_tables(input_json, output_json, img_dir)






## Use Gpt to analyze table
# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('table_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def table_sys_prompt():
    return f'''
Extract structured information from a table, focusing on row-by-row parsing rather than by individual columns. The objective is to create a natural language description for each row that includes the entire row’s context and values, for use in Retrieval-Augmented Generation (RAG) model training.

Output format:

[Table Level]
Extract and structure the following:
- Table Title: Identify the semantic title of the table.
- Table Summary: Provide a concise overview of what the table represents in 2-3 sentences
- Context: Summarize relevant information appearing before or after the table in 1-2 sentences
- Special Notes: Mention any important footnotes, units, or special formatting

[Row Level]
For each row, create a natural language description that:
- Combines column headers with all non-empty cell values in a readable sentence
- Preserves numerical relationships and comparisons
- Includes relevant context from surrounding text
- Maintains semantic relationships between columns
- Uses proper units and formatting from the table

Format each row description as: "Row {{index}}: {{natural language description}}"

Example Output:
[Table Level]
Title: Global Renewable Energy Adoption Rates in 2023 first half
Summary: This table compares renewable energy adoption across countries, showing installation capacity and growth rates from 2020-2023.
Context: The report examines renewable energy growth across major economies. These adoption rates suggest an accelerating transition to clean energy.
Notes: Capacity values are in gigawatts (GW). Growth rates adjusted for seasonal variations. 

[Row Level]
Row 1: China leads renewable energy adoption with 573 GW installed capacity and 12.3% year-over-year growth, having completed most planned installations (A).
Row 2: The United States follows with 325 GW capacity, showing 8.7% growth while several projects remain under development.

Rules:
1. Preserve All Numerical Values: Keep all values and units.
2. Combine Column Data into a Single Description: Parse each row as a whole without splitting by columns.
3. Use Consistent Formatting: Ensure all descriptions follow the same format.
'''


def table_user_prompt(ctx_before: str, ctx_after: str):
    ret = 'Process this table.'
    if ctx_before:
        ret += f'[Context before Table]\n{ctx_before}\n\n'
    if ctx_after:
        ret += f'[Context after Table]\n{ctx_after}\n\n'
    return ret


def extract_table_content(client: OpenAI, image_path: str, ctx_before: str, ctx_after: str) -> str:
    """Extract table content using GPT-4 Vision."""
    base64_image = encode_image_to_base64(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": table_sys_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": table_user_prompt(ctx_before, ctx_after)
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in GPT API call: {e}")
        return ""


def create_chunks(table_content: str, page_num: int, chunk_size: int) -> List[Dict]:
    """Create chunks from table content with context."""
    chunks = []
    table_info, row_info = table_content.split("[Row Level]")
    table_info = table_info.split("[Table Level]")[1].strip()

    # Remove extra whitespace and normalize newlines
    row_info = ' '.join(row_info.strip().split())
    row_info = row_info.replace(' Row ', '\nRow ')
    rows = row_info.split("\n")

    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        chunk_text = table_info + '\n' + '\n'.join(chunk)
        # Create chunk metadata
        chunk_dict = {
            'content': chunk_text,
            'row_count': len(chunk),
            'start_row': i,
            'page_number': page_num
        }
        chunks.append(chunk_dict)
    return chunks


def process_tables(input_json_path: str, output_json_path: str, img_dir: str):
    """Main pipeline to process tables and create chunks."""
    client = OpenAI(
        api_key="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",
    )

    # Read input JSON
    with open(input_json_path, 'r') as f:
        table_data = json.load(f)

    all_chunks = [table_data[0]]  # Add table metadata to chunks

    for item in tqdm(table_data[1:], desc="Processing tables"):
        image_path = os.path.join(img_dir, item['img_path'].split('/')[-1])
        ctx_before = item.get('ctx_before', '')
        ctx_after = item.get('ctx_after', '')
        page_num = item.get('page_num', 0)

        logger.info(f"Processing table from {image_path}")

        while True:
            try:
                # Extract table content
                table_content = extract_table_content(client, image_path, ctx_before, ctx_after)

                # Create chunks
                if table_content:
                    chunks = create_chunks(table_content, page_num, chunk_size=50)
                    all_chunks.extend(chunks)
                break
            except Exception as e:
                # if is RateLimitError error
                if 'RateLimitError' in str(e):
                    time.sleep(10)
                    continue
                logger.error(f"Error processing table: {e}")
                logger.info(f"Skipping table from {image_path}")
                break

    # Save chunks to output JSON
    with open(output_json_path, 'w') as f:
        json.dump(all_chunks, f, indent=2)

    logger.info(f"Processed {len(all_chunks)} chunks saved to {output_json_path}")



if __name__ == '__main__':
    main()
