import re
import sys
import time
import json
import logging
import pandas as pd
# from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

from datetime import datetime
from typing import Dict

# from .apiOllamaManager import ChatManager
from .vllmManager import ChatManager
from .ragManager import RAGManager
from gpu_log import log_gpu_usage
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import FlagLLMReranker



# def truncate_to_token_limit(text, max_tokens=6000):
#     tokens = tokenizer.encode(text)
#     token_count = len(tokens)
    
#     if token_count > max_tokens:
#         truncated_tokens = tokens[:max_tokens]
#         truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
#         print(f"Warning: RAG context truncated from {token_count} to {max_tokens} tokens")
#         return truncated_text
#     return text


def truncate_to_token_limit(text, max_tokens=6000, chars_per_token=3.5):
    estimated_tokens = len(text) // chars_per_token
    if estimated_tokens > max_tokens:
        max_chars = int(max_tokens * chars_per_token)
        truncated_text = text[:max_chars]
        print("-"*50)
        print(f"Warning: Text truncated from ~{estimated_tokens} to {max_tokens} estimated tokens")
        print("-"*50)
        return truncated_text
    return text

def select_most_recent_time(time_info):

    time_info_as_dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in time_info]
    most_recent_date = max(time_info_as_dates)
    
    return most_recent_date.strftime("%Y-%m-%d")
    
def get_rag_content(chat_manager: ChatManager, chunks, rewritten_question: str, query_time: datetime, retriever):
    # print("topk:",topk )
    rag_content = ""
    time_info_list = []

    top_bundle_id = chat_manager.rank_chunk(chunks, rewritten_question, query_time, retriever)

    for bundle_id in top_bundle_id:
        bundle_chunks = [chunk for chunk in chunks if chunk['bundle_id'] == bundle_id]
        bundle_chunk_content = [chunk['page_content'] for chunk in bundle_chunks]
        page_content = " ".join(chunk['page_content'] for chunk in bundle_chunks)
        if len(page_content) < 50:
            continue
        time_info = bundle_chunks[0]['metadata'].get('date_published', None)
        rag_id = bundle_chunks[0]['metadata']['doc_id']

        time_info_list.append(time_info)
        # current_content = f"From {time_info}: {page_content}\n"
        current_content = f"{page_content}"
        rag_content += current_content
        # chat_manager.rag_info.loc[len(chat_manager.rag_info)] = [f"{time_info}", f"{rag_id}",f"{current_content}", bundle_chunks]
        new_row = pd.DataFrame({
            'timeinfo': [f"{time_info}"],
            'rag_id': [f"{rag_id}"],
            'rag_content': f"{current_content}",
            'raw_rag_context': [bundle_chunk_content]
        })
        chat_manager.rag_info = pd.concat([chat_manager.rag_info, new_row], ignore_index=True) 
    return rag_content, time_info_list

class ChatService:

    def __init__(self, config, rag_manager: RAGManager, rerank_topk: int):
        self.api_chat_manager: Dict[str, ChatManager] = {}
        self.rag_manager: RAGManager = rag_manager
        self.base_url: str = config.get('ollama_base_url')
        self.model_name: str = config.get('llm')
        self.rerank_topk = rerank_topk
        
        self.reranker = FlagLLMReranker(config.get('rerank_model'), use_fp16=True) 

        if not self.model_name or not self.base_url:
            logging.error("LLM model name/base_url is not configured.")
            sys.exit(1)
        logging.info(f"Using model: {self.model_name}, URL: {self.base_url}")

    def get_or_create_chat_manager(self, session_id: str) -> ChatManager:
        if session_id not in self.api_chat_manager:
            self.api_chat_manager[session_id] = ChatManager(session_id, self.base_url, self.model_name, self.reranker, chunk_topk = self.rerank_topk)
        return self.api_chat_manager[session_id]
        
        
    
    def generate_response_with_rag(self, question: str, session_id: str, internal_input=None, interrupt_index=None):
        chat_manager = self.get_or_create_chat_manager(session_id)
        lang = '中文' if bool(re.search(r'[\u4e00-\u9fff]', question)) else 'English'
        user_input = question
        qa_history = chat_manager.get_qa_history()
        rewrite_start_time = time.perf_counter()
        rewritten = chat_manager.if_query_rag(user_input, qa_history)
        rewrite_end_time = time.perf_counter()
        logger.info("The time for rewrite: {:.2f}".format(rewrite_end_time-rewrite_start_time))

        
        chat_manager.reset_rag_info()
        answer = ""
        rewritten_q = ""
        if isinstance(rewritten, list):
            hypo_chunk_content = []
            all_retrieved_content = []
            for rewritten_question in rewritten:
                rewritten_q += rewritten_question
                user_input = rewritten_question
        
                rag_context = ""
                query_time = chat_manager.get_query_time(user_input, qa_history)

                if chat_manager.need_rag:
                    log_gpu_usage('rag started')
                    timeinfo_list = []
                    
                    for retriever in self.rag_manager._retrievers:
                        # hyDE rewrite the questions by generating documents
                        hyde_start_time = time.perf_counter()
                        hyde_chunks = chat_manager.generate_hypo_chunks(rewritten_question)
                        hypo_chunk_content.append(hyde_chunks)
                        # hyde_chunks = []
                        logger.info(f"hypo chunks: {hyde_chunks}")
                        logger.info("The time for hyde: {:.2f}".format(time.perf_counter()-hyde_start_time))

                        retriever_content = retriever.invoke(user_input, hyde_chunks)
                        all_retrieved_content.append(retriever_content)
                        rerank_start_time = time.perf_counter()
                        current_context, timeinfo_list = get_rag_content(chat_manager, retriever_content, rewritten_question, query_time, retriever)
                        rerank_end_time = time.perf_counter()
                        logger.info("The time for rerank: {:.2f}".format(rerank_end_time-rerank_start_time))
                        log_gpu_usage('rag finished')
                        rag_context += current_context + '\n'
                        logger.info(f'Input Rag Context is: {rag_context}')

                    used_time = select_most_recent_time(timeinfo_list)
                    chat_manager.add_time_in_sys(used_time)
                response_start_time = time.perf_counter()

                rag_context = truncate_to_token_limit(rag_context)

                response = chat_manager.chat_internal(user_input, rag_context, lang, False, 
                                                    internal_input=internal_input, 
                                                    interrupt_index=interrupt_index)
                response_end_time = time.perf_counter()
                logger.info("The time for response: {:.2f}".format(response_end_time - response_start_time))
                chat_manager.remove_time_in_sys()

                # assert response.status_code == 200
                answer += response.choices[0].message.content

                logger.info(f"Questions: {rewritten_question}")
                logger.info(f"Answer: {response.choices[0].message.content}")

                # answer = response.json()['message']['content']

                chat_manager.save_chat_history(answer)
                chat_manager.add_to_qa_history(user_input, answer)

        if chat_manager.mult_question:
            answer = chat_manager.modify_answer(answer=answer, stream=False, lang=lang)
            logger.info(f"Final answer: {answer}")
        return answer, rag_context, chat_manager.rag_info, rewritten, hypo_chunk_content, all_retrieved_content



    def generate_response_stream(self, question: str, session_id: str, internal_input=None, interrupt_index=None):
        start_time = time.perf_counter()
        chat_manager = self.get_or_create_chat_manager(session_id)
        lang = '中文' if bool(re.search(r'[\u4e00-\u9fff]', question)) else 'English'
        user_input = question
        qa_history = chat_manager.get_qa_history()
        rewritten = chat_manager.if_query_rag(user_input, qa_history)

        chat_manager.reset_rag_info()
        answer = ""
        rewritten_q = ""
        for rewritten_question in rewritten:
            rewritten_q += rewritten_question
            user_input = rewritten_question
            
            rag_context = ""
            
            query_time = chat_manager.get_query_time(user_input, qa_history)
            
            logger.info(user_input)

            if chat_manager.need_rag:
                log_gpu_usage('rag started')
                timeinfo_list = []
                
                for retriever in self.rag_manager._retrievers:
                    # hyDE rewrite the questions by generating documents
                    hyde_chunks = chat_manager.generate_hypo_chunks(rewritten_question)
                    retriever_content = retriever.invoke(user_input, hyde_chunks)
                    current_context, timeinfo_list = get_rag_content(chat_manager, retriever_content, rewritten_question, query_time,retriever)
                    log_gpu_usage('rag finished')
                    rag_context += current_context + '\n'

                used_time = select_most_recent_time(timeinfo_list)
                chat_manager.add_time_in_sys(used_time)

                logger.info(f"RAG context:\n{rag_context}")

            log_gpu_usage('AI speak')
            if not chat_manager.mult_question:
                response = chat_manager.chat_internal(user_input, rag_context, lang, True, 
                                                    internal_input=internal_input, 
                                                    interrupt_index=interrupt_index)
                break
            log_gpu_usage('AI speak finish')

            response = chat_manager.chat_internal(user_input, rag_context, lang, False, 
                                                    internal_input=internal_input, 
                                                    interrupt_index=interrupt_index)
            answer += response.choices[0].message.content

            logger.info(f"Questions: {rewritten_question}")
            logger.info(f"Answer: {response.choices[0].message.content}")
        
        # chat_manager.remove_time_in_sys()
        
        if chat_manager.mult_question:
            response = chat_manager.modify_answer(answer=answer,stream=True)
        first_response_sent = False
        full_response = ""

        for chunk in response:
            try:
                content = chunk.choices[0].delta.content
                if content == "":
                    continue
                if not first_response_sent:
                    response_time = time.perf_counter() - start_time
                    logging.info(f"Time to first response: {response_time:.2f} seconds")
                    first_response_sent = True

                json_data = json.dumps({'response': content, 'general_or_rag': chat_manager.need_rag})
                yield f"data: {json_data}\n\n"
            except json.JSONDecodeError:
                pass
        
        # for line in response.iter_lines():
        #     if line:
        #         try:
        #             json_line = json.loads(line.decode('utf-8'))
        #             if 'message' in json_line and 'content' in json_line['message']:
        #                 content = json_line['message']['content']
        #                 full_response += content
                        
        #                 if not first_response_sent:
        #                     response_time = time.perf_counter() - start_time
        #                     logging.info(f"Time to first response: {response_time:.2f} seconds")
        #                     first_response_sent = True
                        
        #                 json_data = json.dumps({'response': content, 'general_or_rag': chat_manager.need_rag})
        #                 yield f"data: {json_data}\n\n"
        #         except json.JSONDecodeError:
        #             pass

        chat_manager.save_chat_history(full_response)
        chat_manager.add_to_qa_history(rewritten_q, full_response)

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
