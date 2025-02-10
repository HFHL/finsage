import logging
from typing import Dict, List
logger = logging.getLogger(__name__)

import requests
import json
from sseclient import SSEClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd


class ChatManager:
    def __init__(self, session_id, base_url, model_name, reranker, chunk_topk = 5, history_limit=20):
        assert history_limit % 2 == 0, "history_limit must be an even number"
        self.session_id = session_id
        self.base_url = base_url
        self.model_name = model_name
        self.llm = ChatOllama(model=self.model_name)

        #self.rerank_tokenizer = rerank_tokenizer
        #self.rerank_model = rerank_model
        self.reranker = reranker
        self.similar_threshhold = 0.7
        #self.rerank_model = SentenceTransformer(rerank_model_name, trust_remote_code=True)

        self.messages = []               # This is the complete input of current question, including sys_ptompt, question, history_summary, rag content
        self.chat_history = []
        self.all_chat_history = [{
            "role": "system", "content": self._sys_template()
        }]
        self.history_summary = ""
        self.need_rag = False            #only for test
        self.qa_history = [ ]            #This is the history of user and the rag llm
        self.time_info = ""
        self.history_limit = history_limit
        self.chunk_topk = chunk_topk
        self.rag_info = pd.DataFrame({
            'timeinfo': [],
            'rag_id': [],
            'rag_content': []
        })

    @staticmethod
    def _sys_template(lang: str='en'):
        # return """You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        # Your role is to assist users by answering questions and providing detailed information about Lotus's \
        # brand promotion and its famous historical models. You answers questions \
        # based on snippets of text provided in context. Answer should being as concise as possible."""
        return f"""You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        Your role is to assist users by answering questions and providing detailed information about Lotus's brand promotion and its famous historical models. \
        You will also receive background information from an internal human assistant, but this information is only for your understanding to better assist the user. \
        Do not include [Internal Assistant] in your responses. \
        Answer the user's questions naturally like human, do not include bullet point directly, incorporating any useful details from the internal assistant's input without explicitly mentioning them. \
        Be as concise as possible in your responses and provide helpful, relevant information to the user.
        """
        
    @staticmethod
    def _qa_template(question, context, lang):
        if context != "":
            return f"""Use the information provided in the Retrieved Context to answer user's question.  The Latest chunks of the context are more important and should be prioritized in case of conflicting information. If there is a conflict between different chunks, rely on the information from the earlier chunk, as it is considered more reliable. Mention all the details in the retrieved context.
            If you cannot find the answer in the provided context, you need to use your own knowledge to answer.
            Retrieved Context: \n{context}\n
            User input: {question}\n
            Answer the user's questions in {lang}"""
        else:
            #return f"""{question}"""
            return f"""The question might be related to a daily common task, in which case, feel free to answer confidently. 
            However, if you are not quite sure or if the question is related to Lotus (e.g., cars, policies, or financial data), provide a partial answer. You can append: "If you need more detailed information, our human assistant can provide it."
            User input: {question}
            Answer the user's questions in {lang}.
            """
        
    def if_query_rag(self, question, qa_history, max_retry=1):
        prompt_template = """
        You are a smart assistant designed to categorize and rewrite questions. Your task is twofold:

        1. **Determine if the user's question requires information from a specific dataset**:
            - The dataset includes detailed historical and technical data about various car models and electric vehicles, or information on proxy statements and prospectuses. 
            - If the user's question involves details about cars (e.g., engine types, production years, car dimensions), electric vehicles (e.g., Lotus-related data, EV policies), or proxy statements/prospectuses (e.g., financial data, business combination, shareholder voting), categorize the question as requiring the dataset (Answer: YES).
            - If the question is general or not related to these specific datasets (e.g., weather, general knowledge, or unrelated topics), categorize it as not requiring the dataset (Answer: NO).

        2. **Rewrit the question according to the Q&A history**:
            - Rewritten question is in English.
            - If related to previous interactions, it should be rewritten to incorporate the relevant context.
            - If the question is vague or unclear, it should be rewritten for better clarity and understanding.

        Here are some example questions related to the datasets:
        - "What engine was used in the Mark I car?"
        - "Emeya是什么时候推出的?"
        - "How many Mark II cars were built?"
        - "Can you provide the specifications for the Mark VI?"
        - "What are the production years for the Mark VIII?"
        - "Please tell me something about Lotus?"
        - "What are the risk factors listed in the Lotus Tech prospectus?"
        - "Can you tell me about the voting procedures for the extraordinary general meeting in LCAA's proxy statement?"
        - "请给我介绍一下最新的电车" (Tell me about the latest electric cars)
        - "How many Momenta convertible Note has in owership of total shares? "

        Any question that involves details about car models, electric vehicles, or mentions keywords such as Lotus, their specifications, history, or technical data, or that refers to company-related information about Lotus (e.g., company status, financial data, stock listing, etc.), as well as requests for specific information from a business combination, financial data, or legal aspects from a proxy statement or prospectus, should be categorized as requiring the specific dataset (Answer: YES).

        General daily questions might include:
        - "What's the weather like today?"
        - "How do I make a cup of coffee?"
        - "What's the capital of France?"
        - "What time is it?"

        For such questions, the answer should be categorized as not requiring the specific dataset (Answer: NO).

        Here is the Q&A history:
        {qa_history}

        Question: {question}

        Respond in the following format:
        - First line: "YES" or "NO" indicating if the question requires the dataset.
        - Second line: Rewritten question.
        """

        tpl = ChatPromptTemplate.from_template(prompt_template)
        chain = tpl | self.llm
        logger.info(f"Original question: {question}")
        #print("qa_history:", qa_history)
        rewritten_question = question
        for i in range(max_retry):
            response = chain.invoke({"question": question, "qa_history": qa_history})
            #print("The judge:",response)
            response_lines = response.content.strip().split("\n")

            if len(response_lines) == 2:
                self.need_rag = "yes" in response_lines[0].strip().lower()
                rewritten_question = response_lines[1].strip()
                break  
            else:
                print("The format of rag judge output is wrong!!!!")

        logger.info(f"Need RAG: {self.need_rag}")
        logger.info(f"Rewritten question: {rewritten_question}")
        return rewritten_question
    
    def summarize_chat_history(self, chat_history, max_retry=1):
        prompt_template = """
        You are a smart assistant designed to summarize conversation history. Your task is to generate a concise summary that captures the main points and context of the entire conversation, including any retrieved information (RAG content) that was used to provide answers.

        Here is the conversation history:
        {chat_history}

        Please provide a summary that:
        - Clearly represents the topics discussed.
        - Captures any questions, answers, key decisions made during the conversation, and any relevant retrieved information.
        - Maintains the user's original language style and avoids altering or translating any specific parts of the conversation.
        - Is brief but informative enough to understand the context of the discussion.

        Respond with the summarized conversation without any additional explanation or labels.
        If the chat_history is empty, you should just reply no chat history.
        """

        tpl = ChatPromptTemplate.from_template(prompt_template)
        chain = tpl | self.llm
        summary = ""
        #print("chat_history:",chat_history)
        print("Generating summary for conversation history.")
        for i in range(max_retry):
            response = chain.invoke({"chat_history": chat_history})
            summary = response.content.strip()
            if summary:
                break

        return summary

    def rank_chunk(self, chunks: List[Dict], question: str, retriever):
        
        bundle_map = {}
        for idx, chunk in enumerate(chunks):
            bundle_map.setdefault(chunk['bundle_id'], []).append(idx)
            
        pairs = [[question, chunk['page_content']] for chunk in chunks]
        # Batch the pairs for reranker
        BATCH_SIZE = 8
        scores = []
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i:i+BATCH_SIZE]
            batch_scores = self.reranker.compute_score(batch_pairs)
            scores.extend(batch_scores)
        scores = torch.tensor(scores)
        # scores = torch.tensor(self.reranker.compute_score(pairs))

        ranked_indices = torch.argsort(scores, descending=True).tolist()

        # 根据 chunks_num 选择合适数量的 chunk，确保总大小不超过 topk
        selected_indices = []
        current_size = 0
        #只有chunk content的 list
        chunk_content_list = []
        chunk_content_list.extend(chunk['page_content'] for chunk in chunks)

        for idx in ranked_indices:
            logger.info(f"chunk {idx} bundle {chunks[idx]['bundle_id']} score: {scores[idx].item()}")
            bundle_id = chunks[idx]['bundle_id']
            bundle = bundle_map[bundle_id]
            # if bunleid is selected, skip
            if bundle_id in selected_indices or current_size + len(bundle) > self.chunk_topk:
                continue
            # remove the similar chunk
            
            similarity = retriever.compute_similarity(chunk_content_list, selected_indices, idx)
            if torch.any(similarity > self.similar_threshhold):
                print(f"chunk{idx} is skip due to similarity")
                continue
            selected_indices.append(bundle_id)
            current_size += len(bundle)
            
        logger.info(f"reverse ranked bundle indices: {selected_indices[::-1]}")
        torch.cuda.empty_cache()
        return selected_indices[::-1]


    def chat(self, user_input, rag_context='', stream=False):
        user_message = {"role": "user", "content": self._qa_template(user_input, rag_context)}
        self.chat_history.append(user_message)
        self.all_chat_history.append(user_message)
        # print(len(self.chat_history), self.chat_history)
        
        data = {
            "model": self.model_name,
            "messages": self.chat_history,
            "stream": stream
        }
        #import pdb; pdb.set_trace()
        response = requests.post(f"{self.base_url}/api/chat", json=data, stream=True)
        response.raise_for_status()

    def chat_internal(self, user_input, rag_context='', lang: str='en', stream=False, internal_input=None, interrupt_index=None):
        # Handle modification of the previous assistant message if the interrupt index is provided
        if interrupt_index is not None:
            self.modify_previous_assistant_message(interrupt_index)

        # Prepend the internal assistant input to the user message if provided
        if internal_input:
            # Prepend the internal input to the user's message
            user_input = f"[Internal Assistant Information]: {internal_input}\n\nUser Input: {user_input}"

        # Now create the user message and append it to the chat history
        user_message = {"role": "user", "content": self._qa_template(user_input, rag_context, lang)}
        self.chat_history.append(user_message)
        self.all_chat_history.append(user_message)
        
        logger.info(f'should response in {lang}')
        self.messages = [{
            "role": "system", "content": self._sys_template(lang)+self.time_info
        }]

        self.messages.append({"role": "assistant", "content": self.history_summary})
        self.messages.append(user_message)
        #print("THE INPUT:")
        #print(self.messages)
        data = {    
            "model": self.model_name,
            "messages": self.messages,
            "stream": stream
        }

        response = requests.post(f"{self.base_url}/api/chat", json=data, stream=True)
        response.raise_for_status()
        #print("self chat history",self.chat_history)
        return response
    

    def add_time_in_sys(self, used_time):
        """
        Modify the system prompt to let llm reply reference time
        """
        self.time_info = f"\nAt the end of your response, include only one sentence stating that the information is based on knowledge available before {used_time}, and ensure that the language used remains consistent with previous responses."
        """
        for message in self.chat_history:
            if message['role'] == 'system':
                message['content'] += f"\nAt the end of your response, include only one sentence stating that the information is based on knowledge available before {used_time}, and ensure that the language used remains consistent with previous responses."
                break  #
        """
    def remove_time_in_sys(self):
        """
        Remove the line that starts with the specific phrase in the system prompt.
        """
        self.time_info = ""
        """
        for message in self.chat_history:
            if message['role'] == 'system':
                target_phrase = "At the end of your response,"
                # 找到目标短语在文本中的位置
                if target_phrase in message['content']:
                    # 获取短语在内容中的位置
                    index = message['content'].find(target_phrase)
                    # 删除从目标短语开始的所有内容
                    message['content'] = message['content'][:index].rstrip()
                break    
        """
    def add_to_qa_history(self, user_input, llm_response):
        # 将新的问答对添加到qa_history
        self.qa_history.append({
            "user": user_input,
            "llm": llm_response
        })
        # 保证qa_history中最多有5个问答对
        if len(self.qa_history) > 5:
            self.qa_history.pop(0)  # 删除最早的一对

    def get_qa_history(self):
        # 将qa_history格式化为字符串，作为大模型的上下文
        qa_context = ""
        for qa in self.qa_history:
            qa_context += f"User: {qa['user']}\nLLM: {qa['llm']}\n"
        return qa_context
                

    def modify_previous_assistant_message(self, interrupt_index):
        """
        Modify the last assistant message by truncating it after the given interrupt index.
        """
        # Find the previous assistant message (which should be the last assistant response in chat history)
        for message in reversed(self.chat_history):
            if message['role'] == 'assistant':
                # Truncate the assistant's message after the interrupt index
                modified_message = message['content'][:interrupt_index]
                
                # Update the message in chat history
                message['content'] = modified_message
                break  # We only modify the last assistant message

          
    def save_chat_history(self, response):
        assistant_message = {"role": "response", "content": response}
        self.chat_history.append(assistant_message)
        self.all_chat_history.append(assistant_message)
        self._trim_chat_history()

    def _trim_chat_history(self):
        # Keep the system message and the last `self.history_limit` user and assistant messages
        non_system_messages = [msg for msg in self.chat_history if msg['role'] != 'system']
        if len(non_system_messages) > self.history_limit:
            self.chat_history = [self.chat_history[0]] + non_system_messages[-self.history_limit:]

    def get_chat_history(self):
        # 将 chat_history 格式化为字符串，作为大模型的上下文
        chat_context = ""
        for entry in self.chat_history:
            chat_context += f"{entry['role']}: {entry['content']}\n"
        return chat_context

    def get_all_chat_history(self):
        return self.all_chat_history

    def clear_chat_history(self):
        self.chat_history = [self.all_chat_history[0]]

    def reset_rag_info(self):
        """
        reset the rag info after each Q&A
        """
        self.rag_info = pd.DataFrame({
        'timeinfo': [],
        'rag_id': [],
        'rag_content': []
    })