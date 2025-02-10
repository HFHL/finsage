import sys
import os
import time
import yaml
import logging
import json
from tqdm import tqdm


log_file = os.path.join('/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/logs','qa_chunks.log')
logging.basicConfig(
    filemode='w',
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.vllmChatService import ChatService
from utils.ragManager import RAGManager
from gpu_log import log_gpu_usage

TOPK = 10
HYDE = True
LLM_Filtered = True
QUESTION_JSON = "14m.json"
RERANK= False

def write_wrapped_text(file, text, max_line_length=80):
    while text:
        # 写入每行最多 max_line_length 个字符
        file.write(text[:max_line_length] + '\n')
        text = text[max_line_length:]


if __name__ == "__main__":

    log_gpu_usage("Test Start")

    config_path = os.getenv('CONFIG_PATH', os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'config',
        'config_vllm.yaml'
    ))

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # print("Reranker model: ", config['rerank_model'])

    # collections = {'lotus': 10, 'lotus_car_stats': 0, 'lotus_brand_info': 0}
    collections = {'lotus': TOPK}
    rag_manager = RAGManager(config=config, collections=collections)
    log_gpu_usage('Documnets retrievers loaded.')
    chat_service = ChatService(config=config, rag_manager=rag_manager)
    log_gpu_usage('Rerank model loaded.')

    questions_folder_path = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/test_questions"
    questions_file = QUESTION_JSON
    qachunk_folder_path = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks"
    questions_file_path = os.path.join(questions_folder_path, questions_file)

    # Create output directory based on markdown file name
    base_name = os.path.splitext(os.path.basename(questions_file_path))[0]
    DIR_PATH = os.path.join(
    qachunk_folder_path,
    'results_filtered' if LLM_Filtered else 'results_unfiltered',
    f'{base_name}_hyde' if HYDE else f'{base_name}_no_hyde'
    )

    # DIR_PATH = os.path.join(
    # qachunk_folder_path,
    # 'base'
    # )

    # remember to set topk to 40 if enable_hyde is False
    # HYDE = True

    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    # Load questions and answers from JSON file
    with open(questions_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\nProcessing File: ", questions_file_path)
    bad_count = 0

    # for idx, item in tqdm(enumerate(data)):
    for idx, item in enumerate(data):
        # Log the file generation
        # logging.info(f"Generating file: question_{idx+1}.txt")
        # save content and rag_context to file
        with open(os.path.join(DIR_PATH, f'question_{idx+1}.txt'), 'w') as file:
            # 添加文件头部信息
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            file.write(f"Generated Date: {current_time}\n")
            # file.write("Description: database_clean_1107，overlap50个单词，指代词替换.\n")
            # file.write("Each question will be processed with RAG pipeline and show relevant information.\n")
            file.write("=" * 80 + "\n\n")  # 分隔线
            # file.write(f"Rerank model: {config['rerank_model']}\n")
            # file.write("=" * 80 + "\n\n")

            session_id = time.time()
            chat_manager = chat_service.get_or_create_chat_manager(session_id)
            question, expected_answer = item['question'], item['answer']
            
            file.write(f'******* Question {idx+1} *******\n')
            file.write(f'---Question---\n{question}\n\n')

            rewritten_questions = chat_manager.if_query_rag(question, "") 
            #rewritten_question = rewritten_question[0]
            #rewritten_question = item["rewritten"]
            questions = rewritten_questions if isinstance(rewritten_questions, list) else [rewritten_questions]
            
            hyde_chunks = []
            if HYDE:
                hyde_chunks = chat_manager.generate_hypo_chunks("".join(questions))
                
            print("hyde_chunks length:", len(hyde_chunks))
            print("hyde_chunks: ", hyde_chunks)

            all_chunks = []

            for q in questions:
                print(f"Processing question {idx+1}, sub-question: {q}")
                current_chunks = rag_manager._retrievers[0].invoke(q, hyde_chunks)
                all_chunks.extend(current_chunks)

            seen = set()
            chunks = []
            
            for chunk in all_chunks:
                chunk_id = (chunk['page_content'], chunk['metadata']['doc_id'])
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    chunks.append(chunk)

            print(f"Chunks before deduplication: {len(all_chunks)}; Chunks after deduplication: {len(chunks)}")

            effective_chunks = chunks
            
            if LLM_Filtered:
                effective_chunks = []
                for chunk in chunks:
                    # check if the chunk is a inclusive answer for the question or not
                    flag = chat_manager.evaluate_chunk(chunk['page_content'], question, expected_answer)
                    if flag:
                        effective_chunks.append(chunk)

            file.write(f'---Rewritten Question---\n{"".join(questions)}\n\n')

            file.write(f'---Expected Answer---\n')
            write_wrapped_text(file, expected_answer)
            file.write('\n')

            # file.write(f'---Recall---\n{len(chunks)}\n\n---At---\n{len(chunks)}\n\n')
            file.write(f'---Recall---\n{len(effective_chunks)}\n\n---At---\n{len(chunks)}\n\n')

            file.write(f'---Retrieved Chunks by EnsembleRetriever with HyDE---\n' if HYDE 
                       else '---Retrieved Chunks by EnsembleRetriever without HyDE---\n')
            # for chunk in chunks:
            for chunk in effective_chunks:
                file.write(f'{chunk}\n')
            file.write('\n')

        # bad_count += len(chunks) == 0
        bad_count += len(effective_chunks) == 0
    
    logging.warning(f'Bad count: {bad_count} / {len(data)} = {bad_count / len(data)}')
    print("Processed to: ", DIR_PATH)
