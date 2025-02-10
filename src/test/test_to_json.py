import sys
import os
import time
import yaml
import logging
import json
from datetime import datetime

log_file = os.path.join('/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/logs','test_to_json.log')
logging.basicConfig(
    filemode='w',
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.vllmChatService import ChatService
from utils.ragManager import RAGManager
from gpu_log import log_gpu_usage


if __name__ == "__main__":

    log_gpu_usage("Test Start")
    
    config_path = os.getenv('CONFIG_PATH', os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'config',
        'config_vllm.yaml'
    ))
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("Reranker model: ", config['rerank_model'])
    collections = {'lotus': 10}
    rag_manager = RAGManager(config=config, collections=collections)
    log_gpu_usage('Documnets retrievers loaded.')
    chat_service = ChatService(config=config, rag_manager=rag_manager, rerank_topk = 10)
    log_gpu_usage('Rerank model loaded.')
    questions_folder_path = "/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/test_questions"
    questions_file = "75Q.json"
    questions_file_path = os.path.join(questions_folder_path, questions_file)

    # if not os.path.exists(DIR_PATH):
    #     os.makedirs(DIR_PATH)
    
    for i in range(1):
        # output_json = f"/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/scenarios/11/rerank10/hyde_reranker.json"
        output_json = f"/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/scenarios/12/r10k10/hyde_reranker_3.json"
        

        with open(questions_file_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
        
        data = []
        last_output_json = None 
        
        for idx, item in enumerate(question_data):
            question = item['question']
            print(f"Processing: {question}")
            session_id = time.time()

            (answer, rag_context, rag_info, rewritten_question, hypo_chunk_content, 
             all_retrieved_content
            ) = chat_service.generate_response_with_rag(
                question, session_id, internal_input=None, interrupt_index=None)
            
            retrieved_content = []
            # question may return emtpy faiss result
            if all_retrieved_content:
                for item in all_retrieved_content[0]:
                    retrieved_content.append(item['page_content'])
            
            hypo_content = hypo_chunk_content[0] if hypo_chunk_content else None

            new_entry = {
                "question": question,
                "rewritten": rewritten_question if rewritten_question else [],
                "answer": answer,
                "hyde": hypo_content,
                "reranked": rag_info['rag_content'].tolist() if not rag_info.empty else [],
                "reranked_raw": sum(rag_info['raw_rag_context'].tolist(), []) if not rag_info.empty else [],
                "all_retrieved": retrieved_content
            }

            # new_entry = {
            #     "question": question,
            #     "rewritten": rewritten_question if rewritten_question else []
            # }

            data.append(new_entry)

            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


            # if (idx+1) % 4 == 0 or idx == len(question_data) - 1:
            #     start_idx = max(0, idx - len(data) + 1)  # Ensure start index isn't negative
            #     output_json = (f"/root/autodl-tmp/RAG_Agent_vllm_tzh/src/"
            #                    f"test/QAwithChunks/A_35/Q35_{start_idx}-{idx}.json")
            #     last_output_json = output_json

            #     with open(output_json, 'w', encoding='utf-8') as f:
            #         json.dump(data, f, ensure_ascii=False, indent=4)
            #     data = []

        print("-"*50)
        if data:
            print(f"Processed to: {output_json}, at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        chat_service.generate_chat_summary(session_id)

