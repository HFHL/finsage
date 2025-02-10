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
from utils.vllmChatService import ChatService, get_rag_content
from utils.ragManager import RAGManager
from gpu_log import log_gpu_usage

TOPK = 10
HYDE = False
LLM_Filtered = False
QUESTION_JSON = "seedtest.json"
QUESTION_JSON = "75.json"
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

    output_folder_path = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/scenarios/test"
    questions_file_path = os.path.join(questions_folder_path, questions_file)

    # Create output directory based on configuration
    base_name = os.path.splitext(os.path.basename(questions_file_path))[0]
    OUTPUT_JSON = os.path.join(
    output_folder_path,
    f'{base_name}_hyde_rerank_result.json' if HYDE and RERANK else
    f'{base_name}_hyde_result.json' if HYDE and not RERANK else
    f'{base_name}_hyde_result.json' if not HYDE and RERANK else
    f'{base_name}_result.json'
    )
    
    # for i in range(2,4):
    OUTPUT_JSON = f'/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/scenarios/faiss_result/no_hyde/75_faiss_{i}.json'

    # remember to set topk to 40 if enable_hyde is False

    # Load questions and answers from JSON file
    with open(questions_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\nProcessing File: ", questions_file_path)
    bad_count = 0

    # for idx, item in tqdm(enumerate(data)):
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as file:
        json_content = []
        for idx, item in enumerate(data):
            session_id = time.time()
            chat_manager = chat_service.get_or_create_chat_manager(session_id)
            question, expected_answer = item['question'], item['answer']
            
            rewritten_questions = chat_manager.if_query_rag(question, "") 
            # questions = rewritten_questions if isinstance(rewritten_questions, list) else [rewritten_questions]
            print(f"Processing question {idx+1}: {question}, " 
                f"decomposed to total {len(rewritten_questions)} sub-question")
            # if len(questions) == 1:
            #     print(f"Processing quesiton {idx+1}: ", "".join(questions))
            # else:
            #     print(f"Processing quesiton {idx+1} with subquestions: ", " ".join(questions))

            
            
            all_chunks = []
            question_hypo_chunk = []
            for q_idx, q in enumerate(rewritten_questions):
                hyde_chunks = []
                if HYDE:
                    hyde_chunks = chat_manager.generate_hypo_chunks(q)
                    print(f'Q{idx+1}, sub{q_idx+1} Hypo Chunks #: {len(hyde_chunks)}')
                    question_hypo_chunk.append(hyde_chunks)
                current_chunks = rag_manager._retrievers[0].invoke(q, hyde_chunks) # list of chunk dicts
                if RERANK:
                    reranked_chunks = []
                    qa_history = chat_manager.get_qa_history()
                    query_time = chat_manager.get_query_time(q, qa_history)
                    reranked_chunks, time_info = get_rag_content(chat_manager, current_chunks, q, query_time, rag_manager._retrievers[0])
                    effective_chunks.append(reranked_chunks)
                all_chunks.extend(current_chunks)

            seen = set()
            chunks = [] # deduplicated chunks
            
            for chunk in all_chunks:
                # chunk_id = (chunk['page_content'], chunk['metadata']['doc_id'])
                chunk_id = (chunk['metadata']['doc_id'])
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    chunks.append(chunk)
            print(f"Chunks before deduplication: {len(all_chunks)}; Chunks after deduplication: {len(chunks)}")
            print("-" * 50)

            effective_chunks = chunks
            
            if LLM_Filtered:
                effective_chunks = []
                for chunk in chunks:
                    # check if the chunk is a inclusive answer for the question or not
                    flag = chat_manager.evaluate_chunk(chunk['page_content'], question, expected_answer)
                    if flag:
                        effective_chunks.append(chunk)

            # if RERANK:
            #     effective_chunks = []
            #     qa_history = chat_manager.get_qa_history()
            #     for q in questions:
            #         user_input = q
            #         query_time = chat_manager.get_query_time(user_input, qa_history)
            #         reranked_chunks, time_info = get_rag_content(chat_manager, chunks, questions, query_time, rag_manager._retrievers[0])
            #         effective_chunks.append(reranked_chunks)
            
            chunk_content = []
            for effective_chunk in effective_chunks:
                page_content = effective_chunk['page_content']
                chunk_content.append(page_content)

            # entry= {
            #     'question': question,
            #     'rewritten': " ".join(rewritten_questions),
            #     'hypo_chunks': question_hypo_chunk,
            #     'answer': expected_answer,
            #     'content': chunk_content
            # }

            # entry= {
            #     'question': question,
            #     'rewritten': " ".join(rewritten_questions),
            #     'hypo_chunks': question_hypo_chunk,
            #     'content': chunk_content
            # }

            entry= {
                'question': question,
                'rewritten': " ".join(rewritten_questions),
                'answer': expected_answer,
                'content': chunk_content
            }

            # bad_count += len(chunks) == 0
            bad_count += len(effective_chunks) == 0
            json_content.append(entry)
        json.dump(json_content, file, ensure_ascii=False, indent=4)
    
    logging.warning(f'Bad count: {bad_count} / {len(data)} = {bad_count / len(data)}')
    print("Processed to: ", OUTPUT_JSON)
