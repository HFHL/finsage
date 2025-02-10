import sys
import os
import time
import yaml
import logging
import json
from tqdm import tqdm
logging.basicConfig(
    filemode='w',
    filename='eval_rewrite.log',
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.vllmChatService import ChatService
from utils.ragManager import RAGManager
from gpu_log import log_gpu_usage

def write_wrapped_text(file, text, max_line_length=80):
    while text:
        # 写入每行最多 max_line_length 个字符
        file.write(text[:max_line_length] + '\n')
        text = text[max_line_length:]

if __name__ == "__main__":

    log_gpu_usage("Test Start")
    
    config_path = os.getenv('CONFIG_PATH', '../../config/config_vllm.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    collections = {'lotus': 10, 'lotus_car_stats': 0, 'lotus_brand_info': 0}
    rag_manager = RAGManager(config=config, collections=collections)
    log_gpu_usage('Documnets retrievers loaded.')
    chat_service = ChatService(config=config, rag_manager=rag_manager)
    log_gpu_usage('Rerank model loaded.')

    questions_folder_path = "./test_questions/"
    questions_file = "single.json"

    questions_file_path = questions_folder_path + questions_file

    # Create output directory based on markdown file name
    base_name = os.path.splitext(os.path.basename(questions_file_path))[0]
    DIR_PATH = os.path.join(os.path.dirname(questions_file_path), f'{base_name}_eval_rewrite_hyde')


    # remember to set topk to 40 if enable_hyde is False
    enable_hyde = True

    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    # Load questions and answers from JSON file
    with open(questions_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bad_count = 0

    for idx, item in tqdm(enumerate(data)):
        # save content and rag_context to file
        with open(os.path.join(DIR_PATH, f'question_{idx}.txt'), 'w') as file:
            # 添加文件头部信息
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            file.write(f"Generated Date: {current_time}\n")
            # file.write("Description: database_clean_1107，overlap50个单词，指代词替换.\n")
            # file.write("Each question will be processed with RAG pipeline and show relevant information.\n")
            file.write("="*80 + "\n\n")  # 分隔线
            
            session_id = time.time()
            chat_manager = chat_service.get_or_create_chat_manager(session_id)

            question, expected_answer = item['question'], item['answer']

            file.write(f'******* Question {idx} *******\n')
            file.write(f'---Question---\n{question}\n\n')

            rewritten_question = chat_manager.if_query_rag(question, "")
            rewritten_question = rewritten_question[0]

            hyde_chunks = []
            if enable_hyde:
                hyde_chunks = chat_manager.generate_hypo_chunks(rewritten_question)

            chunks = rag_manager._retrievers[0].invoke(rewritten_question, hyde_chunks)

            effective_chunks = []
            for chunk in chunks:
                # check if the chunk is a inclusive answer for the question or not
                flag = chat_manager.evaluate_chunk(chunk['page_content'], question, expected_answer)
                if flag:
                    effective_chunks.append(chunk)

            file.write(f'---Rewritten Question---\n{rewritten_question}\n\n')

            # if show_if_rag:
            #     file.write(f'---Need RAG---\n{need_rag}\n\n')

            # if show_history_summary:
            #     file.write(f'---History Summary---\n')
            #     write_wrapped_text(file, history_summary)
            #     file.write('\n')

            # if show_rag_info:
            #     file.write(f'---RAG Info (DataFrame)---\n{rag_info.to_string(index=False)}\n\n')

            # if show_input:
            #     file.write(f'---Complete Input---\n')
            #     write_wrapped_text(file, ' '.join(str(item) for item in complete_input) + '\n')
            #     file.write('\n')

            file.write(f'---Expected Answer---\n')
            write_wrapped_text(file, expected_answer)
            file.write('\n')

            file.write(f'---Recall---\n{len(effective_chunks)}\n\n---At---\n{len(chunks)}\n\n')
            
            file.write(f'---Effective Chunks---\n')
            for chunk in effective_chunks:
                file.write(f'{chunk}\n')
            file.write('\n')

        bad_count += len(effective_chunks) == 0

    logging.warning(f'Bad count: {bad_count} / {len(data)} = {bad_count / len(data)}')