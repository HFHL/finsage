from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import ast
import requests
import json
from sseclient import SSEClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from gpu_log import log_gpu_usage
from typing import List, Dict, Tuple

class ChatManager:
    def __init__(self, session_id, base_url, model_name, reranker, chunk_topk = 5, history_limit=20):
        assert history_limit % 2 == 0, "history_limit must be an even number"
        self.session_id = session_id
        self.base_url = base_url
        self.model_name = model_name
        # self.llm = ChatOllama(model=self.model_name)
        self.llm = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )

        #self.rerank_tokenizer = rerank_tokenizer
        #self.rerank_model = rerank_model
        self.reranker = reranker
        self.similar_threshhold = 0.9
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
            'rag_content': [],
            'raw_rag_context': []
        })
        self.mult_question = False

    @staticmethod
    def _sys_template(lang: str='en'):
        # return """You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        # Your role is to assist users by answering questions and providing detailed information about Lotus's \
        # brand promotion and its famous historical models. You answers questions \
        # based on snippets of text provided in context. Answer should being as concise as possible."""
        return f"""You are Colin, an LLM-driven guide for Lotus Starlight Avenue.
Your role is to assist users by answering questions related to Lotus’s brand promotion and its famous historical models.
You will receive background information from an internal human assistant for context, but do not include this information directly in your responses.
Do not include [Internal Assistant] in your responses.
Answer the user's questions naturally like human, do not include bullet point directly, avoiding unnecessary details that are not closely related to the query.
Incorporating any useful details from the internal assistant's input without explicitly mentioning them.
Focus on providing helpful, relevant information without over-explaining.
Do not provide outdated information.
DO NOT INCLUDE ANY DETAILS THAT ARE NOT DIRECTLY RELATED TO THE QUESTION.
        """
        
    @staticmethod
    def _qa_template(question, context, lang):
        if context != "":
            # return f"""Use the information provided in the Retrieved Context to answer user's question.  The earlier chunks of the context are more important and should be prioritized in case of conflicting information. If there is a conflict between different chunks, rely on the information from the earlier chunk, as it is considered more reliable. 
            # If you cannot find the answer in the provided context, you need to use your own knowledge to answer.
            # Retrieved Context: \n{context}\n
            # User input: {question}\n
            # Answer the user's questions in {lang}"""
        
            # return f"""Use the information provided in the Retrieved Context to answer user's question.  The Latest chunks of the context are more important and should be prioritized in case of conflicting information. If there is a conflict between different chunks, rely on the information from the earlier chunk, as it is considered more reliable. Mention all the details in the retrieved context.
            # Make sure not to mix the information from different paragraphs in the same sentence and separate in different sentences.
            # If you cannot find the answer in the provided context, you need to use your own knowledge to answer.
            # Retrieved Context: \n{context}\n
            # User input: {question}\n
            # Answer the user's questions in {lang}"""

            # return f"""
            # Use the information provided in the Retrieved Context to answer the user's question. Prioritize details from the latest chunks, as they are more likely to be up-to-date. If there is a conflict between different chunks, rely on the information from the latest chunk. Mention all the relevant details in the retrieved context.
            # Do not combine or merge information from different chunks into a single sentence. Ensure that information from each chunk is presented independently, without mixing with details from other chunks.
            # If the answer cannot be determined from the retrieved context, use your own knowledge to answer.
            # Retrieved Context: \n{context}\n
            # User input: {question}\n
            # Answer the user's questions in {lang}"""
            

            return f"""
            Use the information provided in the Retrieved Context to answer the user's question. 
            - Each chunk begins with "From" and a date, for example: From 2024-09-19
            - Prioritize details from the latest chunks, as they are more likely to be up-to-date. If there is a conflict between different chunks, rely on the information from the latest chunk.
            - Do not combine or merge information from different chunks into a single sentence. Ensure that answer presents the information from each chunk independently, without mixing with details from other chunks.
            - If the answer cannot be determined from the retrieved context, use your own knowledge to answer.
            - DO NOT INCLUDE ANY DETAILS THAT ARE NOT DIRECTLY RELATED TO THE QUESTION.
            - Break down your answer by each mentioned category/dimension in the question, addressing each data point separately. If specific information is not available for any category, explicitly state 'no information available' for that item. For example, Break down 2024 delivery volumes by region (China/US/Europe/Others) and quarter (Q1-Q3). State 'no info' for missing data

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
        You are a smart assistant designed to categorize and rewrite questions. Your task contains 3 steps:

        1. **Determine if the user's question requires information from a specific dataset**:
            - The dataset includes detailed historical and technical data about various car models and electric vehicles, or information on proxy statements and prospectuses. 
            - If the user's question involves details about cars (e.g., engine types, production years, car dimensions), electric vehicles (e.g., Lotus-related data, EV policies), transactions with other company, or proxy statements/prospectuses (e.g., financial data, business combination, shareholder voting), categorize the question as requiring the dataset (Answer: YES).
            - If user's question involves "company", it always means "Lotus Tech", and the Chinese name for 'Lotus Technology' is "路特斯科技"  (Answer: YES).

            Any question that involves details about car models, electric vehicles, or mentions keywords such as Lotus, their specifications, history, or technical data, or that refers to company-related information about Lotus (e.g., company status, financial data, stock listing, etc.), as well as requests for specific information from a business combination, financial data, or legal aspects from a proxy statement or prospectus, should be categorized as requiring the specific dataset (Answer: YES).
            Here are some example questions related to the datasets:
            "What engine was used in the Mark I car?"
            "Emeya是什么时候推出的?"
            "How many Mark II cars were built?"
            "Can you provide the specifications for the Mark VI?"
            "What are the production years for the Mark VIII?"
            "Please tell me something about Lotus?"
            "What are the risk factors listed in the Lotus Tech prospectus?"
            "Can you tell me about the voting procedures for the extraordinary general meeting in LCAA's proxy statement?"
            "请给我介绍一下最新的电车" (Tell me about the latest electric cars)
            "How many Momenta convertible Note has in owership of total shares? "
            "介绍一下Kershaw Health Limited"
            "简单描述一下Meritz的交易"

            - If the question is general or not related to these specific datasets (e.g., weather, general knowledge, or unrelated topics), categorize it as not requiring the dataset (Answer: NO).
            For such questions, the answer should be categorized as not requiring the specific dataset (Answer: NO).
            General daily questions might include:
                "What's the weather like today?"
                "How do I make a cup of coffee?"
                "What's the capital of France?"
                "What time is it?"


        2. **Determine if the user's question includes more than one subquestions**:
            - If the input contains more than one distinct question, respond with "YES".
            Split user's question into individual sub-questions and do step 3 for each individual sub-questions.
            The output for step 3 in this case should be a string list that contains all sub-questions after rewriting.

            - If the input contains only one question, respond with "NO".
            Go to step 3 and rewrite the question.
            The output for step 3 in this case should just be a string list with 1 element (the rewritten question).

        3. **Rewrite the question according to the Q&A history**:
            - Rewritten question is in English.
            - If related to previous interactions, it should be rewritten to incorporate the relevant context.
            - If the question is vague, unclear, or lacks a specific subject, it should be rewritten for better clarity and understanding. 
            - If no specific subject is mentioned or can be derived from the history, include "Lotus Technology" in the rewritten question as the default subject.

        4. **Make each rewritten question self-contained**:
            - Each rewritten question must include the complete subject/context even if it's a sub-question
            - Avoid using pronouns (it, they, these) in follow-up sub-questions
            - Always repeat the full subject name in each sub-question

        Here is the Q&A history:
        {qa_history}

        Question: {question}

        Respond in the following format:
        - First line: "YES" or "NO" indicating if the question requires the dataset.
        - Second line: "YES" or "NO" indicating if use's input includes multiple questions.
        - Third line: Rewritten question is a string list with every question is enclosed in double quotes (") and separated by commas.
        For example: ["question1", "question2", "question3"].
        """

        logger.info(f"Original question: {question}")
        #print("qa_history:", qa_history)
        rewritten_question = question
        for i in range(max_retry):
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_template.format(qa_history=qa_history, question=question)},
                    {"role": "user", "content": question}
                ],
                temperature=0,
                top_p=0.8,
                stream=False
            )
            response_lines = completion.choices[0].message.content.strip().split("\n")

            if len(response_lines) == 3:
                self.need_rag = "yes" in response_lines[0].strip().lower()
                self.mult_question = "yes" in response_lines[1].strip().lower()
                rewritten_question = ast.literal_eval(response_lines[2].strip())
                break
            else:
                logger.info("The format of rag judge output is wrong!!!!")

        logger.info(f"Need RAG: {self.need_rag}")
        logger.info(f"Multiple Questions: {self.mult_question}")
        logger.info(f"Rewritten question: {rewritten_question}")
        return rewritten_question
    
    def get_query_time(self, question, chat_history, max_retry=1):
        prompt_template = """
        You are a smart assistant designed to determine the time reference for a conversation. Your task is to identify the relevant date or time based on the user’s question and the context of the conversation. If no specific time is mentioned, use the current date as the default reference time.
        
        Instructions:
        1.	Analyze the conversation history and the question to identify any explicit or implied time reference.
        2.	If no time reference is present, default to the current date.

        Input Details:
	    •	Conversation History: {chat_history}
	    •	Question: {question}
	    •	Current Date: {now_date}

        Response Format:
	    •	Provide ONLY the reference time in the format “YYYY-MM-DD” (e.g., “2022-01-01”).
        """
        for _ in range(max_retry):
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_template.format(chat_history=chat_history, question=question, now_date=datetime.now().strftime("%Y-%m-%d"))},
                    {"role": "user", "content": question}
                ],
                temperature=0,
                top_p=0.8,
                stream=False
            )
            time_info = completion.choices[0].message.content.strip()
            # if time_info can be converted to a valid date, then break
            try:
                query_time = datetime.strptime(time_info, "%Y-%m-%d")
                break
            except:
                logger.info("The format of time query output is wrong!!!!")
        logger.info(f"Query Time: {query_time}")
        return query_time

    def generate_hypo_chunks(self, question: str) -> List[str]:

        # client = OpenAI(api_key="sk-4768b45eb65f407790a619db44c37f32", base_url="https://api.deepseek.com")
        # model_name = "deepseek-chat"

        client = OpenAI(api_key="sk-4FuXXN5mrWi3jadx3f14E40f0f114eF4BbBaF00c7a58D303", base_url='https://az.gptplus5.com/v1')
        model_name = "gpt-4o"

        hyde_prompt = f"""
        You are a highly intelligent assistant tasked with assisting in the retrieval of real documents. Given the user’s question below, create three hypothetical answers that are contextually relevant and could serve as a useful basis for retrieving real documents. Each answer should be detailed, informative, and approximately 50 words in length. The answers should address different aspects of the user’s query, be logically structured, and provide enough variation in wording and sentence structure to guide the retrieval of actual documents.

        If the query asks for specific data (e.g., sales figures, employee statistics, financials, etc.), include an answer formatted as follows:

            [Table Level]
            •	Table Title: [Title]
            •	Table Summary: [A brief description of the table content, what data it represents, and any relevant timeframes or categories.]
            •	Context: [Explanation of the data’s context or significance, why it’s important, and how it can be used.]
            •	Special Notes: [Any additional details or important points about the data.]

            [Row Level]
            •	Row 1: [Data]
            •	Row 2: [Data]

            Response format:

        ANSWER: [Answer content related to the query]

        ANSWER: [Answer content related to the query]

        ANSWER: [Answer content related to the query]
                """

        # completion = self.llm.chat.completions.create(
        try_cnt = 3
        while try_cnt > 0:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": hyde_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0,
                    top_p=1,
                    seed=42,
                    stream=False
                )
                print(f'Generating Hypo Chunks using [{completion.model}]')
                hypothetical_context = completion.choices[0].message.content
                chunk_list = [chunk.strip() for chunk in hypothetical_context.split("ANSWER:")[1:]]
                return chunk_list
            except Exception as e:
                print(f"Error while generating hypothetical chunks: {e}")
            try_cnt -= 1
        logger.info(f"Question: {question} - Failed to generate Hypo Chunks after 3 retries")
        print(f"Failed to generate Hypo Chunks for {question} after 3 retries")
        return []
    
    def summarize_chat_history(self, chat_history, max_retry=1):
        # prompt_template = """
        # You are a smart assistant designed to summarize conversation history. Your task is to generate a concise summary that captures the main points and context of the entire conversation, including any retrieved information (RAG content) that was used to provide answers.

        # Here is the conversation history:
        # {chat_history}

        # Please provide a summary that:
        # - Clearly represents the topics discussed.
        # - Captures any questions, answers, key decisions made during the conversation, and any relevant retrieved information.
        # - Maintains the user's original language style and avoids altering or translating any specific parts of the conversation.
        # - Is brief but informative enough to understand the context of the discussion.

        # Respond with the summarized conversation without any additional explanation or labels.
        # If the chat_history is empty, you should just reply no chat history.
        # """

        prompt_template = """
        You are a smart assistant designed to summarize conversation history. 
        Your task is to generate a concise summary that captures the main points and context of the entire conversation, including any retrieved information (RAG content) that was used to provide answers.
        For the retrieved information paragraphs, avoid mixing the information from different paragraphs into one single sentence.
        
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

        summary = ""
        #print("chat_history:",chat_history)
        print("Generating summary for conversation history.")
        for i in range(max_retry):
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_template.format(chat_history=chat_history)},
                    {"role": "user", "content": "Summarize the conversation history."}
                ],
                temperature=0,
                top_p=0.8,
                stream=False,
            )
            summary = completion.choices[0].message.content.strip()
            if summary:
                break

        return summary


    def modify_answer(self, answer, stream, lang, max_retry=1):
        prompt_template = f"""
        Instructions:
        Based on the answer provided in the input, rewrite the answer clearly and concisely. Ensure that the response is free of repeated information.

        For repeated information, include it only once in the response.
        If there is additional information (such as services or other offerings), present it separately but avoid repeating content.

        Input Details:
	    •	Answer: {answer}

        Respond with the rewritten answer.
        Answer the user's questions in {lang}
        """


        for i in range(max_retry):
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_template.format(answer=answer)},
                    {"role": "user", "content": "Rewrite the answer without repeated information"}
                ],
                temperature=0,
                top_p=0.8,
                stream=stream
            )
            if not stream:
                rewritten_answer = completion.choices[0].message.content.strip()
                if rewritten_answer:
                    break
            else:
                return completion

        return rewritten_answer
    
    def evaluate(self, answer, expected_answer) -> Tuple[float, str]:
        # Use LLM to evaluate the answer, return the score [0-1], 1 means the answer totally match the expected answer and include all the information. 0 means totally different.
        prompt_template = f"""
        You are a smart assistant designed to evaluate answers provided. Your task is to compare the given answer with the expected answer and assign a score ranging from 0 to 1 based on its relevance and accuracy. The evaluation must consider whether the given answer includes all the numbers and points in the expected answer.
        • A score of 1 indicates that the given answer includes all the numbers and points in the expected answer.
        • A score of 0 indicates that the given answer is irrelevant, inaccurate, or does not include any of the key information from the expected answer.
        • Scores between 0 and 1 reflect partial relevance and accuracy, based on how much of the expected answer’s information is included.

        In addition to the score, provide a brief explanation of the reasoning behind the assigned score.

        Output your response in the following format:
        
        Score: [score]
        Reason: [brief explanation, focusing on whether the given answer includes all or part of the expected answer and the overall relevance and accuracy.]
        """

        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": f"Answer: {answer}\nExpected Answer: {expected_answer}"}
            ],
            temperature=0,
            top_p=0.8,
            stream=False
        )
        resp = completion.choices[0].message.content.strip()
        score = float(resp.split("Score:")[1].split("Reason:")[0].strip())
        reason = resp.split("Reason:")[1].strip()
        return score, reason

    def evaluate_chunk(self, chunk: str, question: str, exp_answer: str) -> bool:
        # Use LLM to evaluate the chunk, return True if the chunk is inclusive to get the expected answer for the question, otherwise return False
        prompt_template = """
        You are a smart assistant whose task is to determine whether the provided chunks of text are relevant for answering the 'Question', and whether they contain one OR more key information necessary to produce the 'Expected Answer'

        Criteria:
        1. Consider the overall context and how chunks may complement each other to form a complete answer.
        2. A chunk should be marked as relevant if it:
        - Contains direct information needed for the answer
        - Answers part (aspect) of the question
        3. For questions requiring multiple aspects, mark chunks as relevant if they address any of:
        - Financial metrics 
        - Strategic planning 
        - Business positioning 
        - Operational aspects 
        - Future outlook 
        - Historical context 
        - Industry relationships 

        Response format:
        Relevance: [YES or NO]
        Reason: [One sentence explains why this chunk contributes to the answer or why it doesn't]
        """

        resp = ""
        try_cnt = 3
        while try_cnt > 0 and (resp != "YES" and resp != "NO"):
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_template},
                    {
                        "role": "user", 
                        "content": f"Question: {question}\nExpected Answer: {exp_answer}\nChunk: {chunk}"
                    }
                ],
                temperature=0,
                top_p=0.8,
                stream=False
            )
            resp = completion.choices[0].message.content.strip()
            flag = resp.split("Relevance:")[1].split("Reason:")[0].strip()
            reason = resp.split("Reason:")[1].strip()
            try_cnt -= 1

        print(f"Question: {question}\nExpected Answer: {exp_answer}\nChunk: {chunk}\nResponse: {flag}\nReason: {reason}")
        return flag == "YES"


    def rank_chunk(self, chunks: List[Dict], question: str, query_time: datetime, retriever):
        
        bundle_map = {}
        for idx, chunk in enumerate(chunks):
            bundle_map.setdefault(chunk['bundle_id'], []).append(idx)

        pairs = [[question, chunk['page_content']] for chunk in chunks]
        # Batch the pairs for reranker
        BATCH_SIZE = 8
        reranker_scores = []
        time_scores = []

        #只有chunk content的 list
        chunk_content_list = []
        chunk_content_list.extend(chunk['page_content'] for chunk in chunks)

        for chunk in chunks:
            # time score = max(0, 1 - |query reference date - date of chunk| / 365)
            score = abs((query_time - datetime.strptime(chunk['metadata']['date_published'], "%Y-%m-%d")).days)
            score = max(0, 1 - score / 365)
            time_scores.append(score)

        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i:i+BATCH_SIZE]
            batch_scores = self.reranker.compute_score(batch_pairs)
            reranker_scores.extend(batch_scores)
        
        reranker_scores = torch.tensor(reranker_scores)
        time_scores = torch.tensor(time_scores)
        scores = reranker_scores + time_scores

        ranked_indices = torch.argsort(scores, descending=True).tolist()

        # 根据 chunks_num 选择合适数量的 chunk，确保总大小不超过 topk
        selected_indices = []
        current_size = 0
        similar_mtx = retriever.compute_similarity_mtx(chunk_content_list)

        for idx in ranked_indices:
            logger.info(f"chunk {idx} bundle {chunks[idx]['bundle_id']} score: {scores[idx].item()}")
            bundle_id = chunks[idx]['bundle_id']
            bundle = bundle_map[bundle_id]
            # if bunleid is selected, skip
            if bundle_id in selected_indices or current_size + len(bundle) > self.chunk_topk:
                continue

            # remove similar chunks
            # similarity = retriever.compute_similarity(chunk_content_list, selected_indices, idx)
            # if torch.any(similarity > self.similar_threshhold):
            #     print(f"chunk{idx} is skip due to similarity")
            #     continue
            if torch.any(similar_mtx[idx, selected_indices] > self.similar_threshhold):
                logger.info(f"chunk{idx} is skip due to similarity")
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
        
        print('should response in ', lang)
        self.messages = [{
            "role": "system", "content": self._sys_template(lang)+self.time_info
        }]

        self.messages.append({"role": "assistant", "content": self.history_summary})
        self.messages.append(user_message)

        log_gpu_usage('AI speak start')

        # client = OpenAI(api_key="sk-4FuXXN5mrWi3jadx3f14E40f0f114eF4BbBaF00c7a58D303", base_url='https://az.gptplus5.com/v1')
        # model_name = "gpt-4o"

        # client = OpenAI(api_key="sk-4768b45eb65f407790a619db44c37f32", base_url="https://api.deepseek.com")
        # model_name = "deepseek-chat"
        
        
        response = self.llm.chat.completions.create(
            messages=self.messages,
            model=self.model_name,
            stream=stream,
            temperature=0.4,
            top_p=0.8,
        )
        return response


        # print(f"Answering using [{model_name}]")
        # try_cnt = 3
        # while try_cnt > 0:
        #     try:
        #         response = client.chat.completions.create(
        #             messages=self.messages,
        #             model=model_name,
        #             stream=stream,
        #             temperature=0.4,
        #             top_p=0.8
        #         )
        #         return response
        #     except Exception as e:
        #         logger.info(f"Error while Answering question; Error: {e};\nCurrent user_input: {user_input}")
        #         print(f"Error while Answering question; Error: {e}, {4-try_cnt}th retry")
        #     try_cnt -= 1
        # logger.info(f"Error while Answering question after 3 retries;\nCurrent user_input: {user_input}")
        # log_gpu_usage('AI speak finish')
        # #print("self chat history",self.chat_history)
        # print(f"failed to answer: {user_input}")
        # return []
    

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
