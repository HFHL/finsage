import sys
import os
import time
import yaml
import logging
import json
logging.basicConfig(
    filemode='w',
    filename='chat_service_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.vllmChatService import ChatService
from utils.ragManager import RAGManager
from gpu_log import log_gpu_usage

def read_questions_from_md(md_file_path):
    questions = []
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                question = line.strip()
                if question:
                    questions.append(question)
    except FileNotFoundError:
        print(f"'{md_file_path}' not found")
    except Exception as e:
        print(f"Error：{e}")
    return questions

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
    questions_file = "eval_rewrite.json"

    questions_file_path = questions_folder_path + questions_file
    # questions = read_questions_from_md(questions_file_path)

    # Create output directory based on markdown file name
    base_name = os.path.splitext(os.path.basename(questions_file_path))[0]
    DIR_PATH = os.path.join(os.path.dirname(questions_file_path), f'{base_name}_output_no_hyde')


    BATCH_SIZE = 1
    show_rag_info = True
    show_history_summary = True
    show_rewritten_question = True
    show_if_rag = False
    show_input = False
    show_total_input = False
    judge_answer = True


    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    # Load questions and answers from JSON file
    with open(questions_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_questions = len(data)
    sum_score = 0

    for i in range(0, total_questions, BATCH_SIZE):
        batch_questions = data[i:i+BATCH_SIZE]
        # save content and rag_context to file
        with open(os.path.join(DIR_PATH, f'question_{i}_{i+BATCH_SIZE-1}.txt'), 'w') as file:
            # 添加文件头部信息
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            file.write(f"Generated Date: {current_time}\n")
            file.write("Description: database_clean_1107，overlap50个单词，指代词替换.\n")
            file.write("Each question will be processed with RAG pipeline and show relevant information.\n")
            file.write("="*80 + "\n\n")  # 分隔线
            
            session_id = time.time()
            for idx, item in enumerate(batch_questions):
                question = item['question']
                expected_answer = item['answer']
                file.write(f'******* Question {idx} *******\n')
                file.write(f'---Question---\n{question}\n\n')
                
                answer, rag_context, rag_info, rewritten_question = chat_service.generate_response_with_rag(question, session_id, internal_input=None, interrupt_index=None)

                history_summary, complete_input, need_rag = chat_service.get_test_info(session_id)

                if show_rewritten_question:
                    file.write(f'---Rewritten Question---\n{rewritten_question}\n\n')

                if show_if_rag:
                    file.write(f'---Need RAG---\n{need_rag}\n\n')

                if show_history_summary:
                    file.write(f'---History Summary---\n')
                    write_wrapped_text(file, history_summary)
                    file.write('\n')

                if show_rag_info:
                    file.write(f'---RAG Info (DataFrame)---\n{rag_info.to_string(index=False)}\n\n')

                if show_input:
                    file.write(f'---Complete Input---\n')
                    write_wrapped_text(file, ' '.join(str(item) for item in complete_input) + '\n')
                    file.write('\n')

                file.write(f'---Answer---\n')
                write_wrapped_text(file, answer)
                file.write('\n')

                if judge_answer:
                    # Compare the generated answer with the expected answer
                    judge_score, reason = chat_service.api_chat_manager[session_id].evaluate(answer, expected_answer)
                    sum_score += judge_score

                    file.write(f'---Expected Answer---\n')
                    write_wrapped_text(file, expected_answer)
                    file.write('\n')
                    file.write(f'---Score---\n')
                    file.write(f'{judge_score}\n\n')
                    file.write(f'---Reason---\n')
                    write_wrapped_text(file, reason)
                    file.write('\n')

                chat_service.generate_chat_summary(session_id)
            chat_service.api_chat_manager[session_id].clear_chat_history()

    if judge_answer:
        # Calculate and print the accuracy score
        accuracy = sum_score / total_questions
        print(f'Average Score: {accuracy:.2f}')