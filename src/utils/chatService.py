import re
import sys
import time
import json
import logging
logger = logging.getLogger(__name__)

from datetime import datetime
from typing import Dict

from .apiOllamaManager import ChatManager
from .ragManager import RAGManager
from gpu_log import log_gpu_usage
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import FlagLLMReranker


def select_most_recent_time(time_info):

    time_info_as_dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in time_info]
    most_recent_date = max(time_info_as_dates)
    
    return most_recent_date.strftime("%Y-%m-%d")
    
def get_rag_content(chat_manager: ChatManager, chunks, rewritten_question: str, retriever):
    # print("topk:",topk )
    rag_content = ""
    time_info_list = []

    start_time = time.time()
    top_bundle_id = chat_manager.rank_chunk(chunks, rewritten_question, retriever)
    elapsed_time = time.time() - start_time
    print("The time for rerank:",elapsed_time)

    for bundle_id in top_bundle_id:
        bundle_chunks = [chunk for chunk in chunks if chunk['bundle_id'] == bundle_id]
        page_content = " ".join(chunk['page_content'] for chunk in bundle_chunks)
        if len(page_content) < 50:
            continue
        time_info = bundle_chunks[0]['metadata'].get('date_published', None)
        rag_id = bundle_chunks[0]['metadata']['doc_id']

        time_info_list.append(time_info)
        current_content = f"From {time_info}: {page_content}\n"
        rag_content += current_content
        chat_manager.rag_info.loc[len(chat_manager.rag_info)] = [f"{time_info}", f"{rag_id}",f"{current_content}" ] 
    return rag_content, time_info_list

class ChatService:

    def __init__(self, config, rag_manager: RAGManager):
        self.api_chat_manager: Dict[str, ChatManager] = {}
        self.rag_manager: RAGManager = rag_manager
        self.base_url: str = config.get('ollama_base_url')
        self.model_name: str = config.get('llm')
        
        self.reranker = FlagLLMReranker(config.get('rerank_model'), use_fp16=True) 

        if not self.model_name or not self.base_url:
            logging.error("LLM model name/base_url is not configured.")
            sys.exit(1)
        logging.info(f"Using model: {self.model_name}, URL: {self.base_url}")

    def get_or_create_chat_manager(self, session_id: str) -> ChatManager:
        if session_id not in self.api_chat_manager:
            self.api_chat_manager[session_id] = ChatManager(session_id, self.base_url, self.model_name, self.reranker)
        return self.api_chat_manager[session_id]
        
    def generate_response_with_rag(self, question: str, session_id: str, internal_input=None, interrupt_index=None):
        chat_manager = self.get_or_create_chat_manager(session_id)
        lang = '中文' if bool(re.search(r'[\u4e00-\u9fff]', question)) else 'English'
        user_input = question
        qa_history = chat_manager.get_qa_history()
        rewrite_start_time = time.perf_counter()
        rewritten_question = chat_manager.if_query_rag(user_input, qa_history)
        rewrite_end_time = time.perf_counter()
        logger.info("The time for rewrite: {:.2f}".format(rewrite_end_time-rewrite_start_time))
        user_input = rewritten_question
        use_time = "no time"
        rag_context = ""
        chat_manager.reset_rag_info()

        if chat_manager.need_rag:
            log_gpu_usage('rag started')
            timeinfo_list = []
            
            for retriever in self.rag_manager._retrievers:
                chunks = retriever.invoke(user_input)
                rerank_start_time = time.perf_counter()
                current_context, timeinfo_list = get_rag_content(chat_manager, chunks, rewritten_question, retriever)
                rerank_end_time = time.perf_counter()
                logger.info("The time for rerank: {:.2f}".format(rerank_end_time-rerank_start_time))
                log_gpu_usage('rag finished')
                rag_context += current_context + '\n'

            if use_time == "no time":
                used_time = select_most_recent_time(timeinfo_list)
                chat_manager.add_time_in_sys(used_time)
        response_start_time = time.perf_counter()

        response = chat_manager.chat_internal(user_input, rag_context, lang, False, 
                                            internal_input=internal_input, 
                                            interrupt_index=interrupt_index)
        response_end_time = time.perf_counter()
        logger.info("The time for response: {:.2f}".format(response_end_time - response_start_time))
        chat_manager.remove_time_in_sys()

        assert response.status_code == 200

        answer = response.json()['message']['content']

        chat_manager.save_chat_history(answer)
        chat_manager.add_to_qa_history(user_input, answer)
        return answer, rag_context, chat_manager.rag_info, rewritten_question



    def generate_response_stream(self, question: str, session_id: str, internal_input=None, interrupt_index=None):
        start_time = time.perf_counter()
        chat_manager = self.get_or_create_chat_manager(session_id)
        
        user_input = question
        qa_history = chat_manager.get_qa_history()
        rewritten_question = chat_manager.if_query_rag(user_input, qa_history)
        user_input = rewritten_question
        lang = '中文' if bool(re.search(r'[\u4e00-\u9fff]', question)) else 'English'
        use_time = "no time"
        rag_context = ""
        chat_manager.reset_rag_info()
        
        logger.info(user_input)

        if chat_manager.need_rag:
            log_gpu_usage('rag started')
            timeinfo_list = []
            
            for retriever in self.rag_manager._retrievers:
                retriever_content = retriever.invoke(user_input)
                current_context, timeinfo_list = get_rag_content(chat_manager, retriever_content, rewritten_question, retriever)
                log_gpu_usage('rag finished')
                rag_context += current_context + '\n'

            if use_time == "no time":
                used_time = select_most_recent_time(timeinfo_list)
                chat_manager.add_time_in_sys(used_time)

            logger.info(f"RAG context:\n{rag_context}")

        log_gpu_usage('AI speak')
        response = chat_manager.chat_internal(user_input, rag_context, lang, True, 
                                            internal_input=internal_input, 
                                            interrupt_index=interrupt_index)
        log_gpu_usage('AI speak finish')
        
        chat_manager.remove_time_in_sys()
        
        first_response_sent = False
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode('utf-8'))
                    if 'message' in json_line and 'content' in json_line['message']:
                        content = json_line['message']['content']
                        full_response += content
                        
                        if not first_response_sent:
                            response_time = time.perf_counter() - start_time
                            logging.info(f"Time to first response: {response_time:.2f} seconds")
                            first_response_sent = True
                        
                        json_data = json.dumps({'response': content, 'general_or_rag': chat_manager.need_rag})
                        yield f"data: {json_data}\n\n"
                except json.JSONDecodeError:
                    pass

        chat_manager.save_chat_history(full_response)
        chat_manager.add_to_qa_history(user_input, full_response)

        self.generate_chat_summary(session_id)

    def generate_chat_summary(self, session_id: str):
        chat_manager = self.get_or_create_chat_manager(session_id)
        try:
            qa_history = chat_manager.get_qa_history()
            #chat_history = chat_manager.get_chat_history()
            #print("--------------")
            #print("Here is the chat_history:",chat_history)
            #print("--------------")
            chat_manager.history_summary = chat_manager.summarize_chat_history(qa_history)
            
            #print(f"Conversation Summary: {chat_manager.history_summary}")
        except Exception as e:
            logging.error(f"An error occurred while generating summary: {str(e)}")

    def get_test_info(self, session_id: str):
        chat_manager = self.get_or_create_chat_manager(session_id)
        
        return chat_manager.history_summary, chat_manager.messages, chat_manager.need_rag
