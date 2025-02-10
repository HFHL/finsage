import requests
import json
from sseclient import SSEClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


class ChatManager:
    def __init__(self, session_id, base_url, model_name, history_limit=20):
        assert history_limit % 2 == 0, "history_limit must be an even number"
        self.session_id = session_id
        self.base_url = base_url
        self.model_name = model_name
        self.llm = ChatOllama(model=self.model_name)
        self.chat_history = [{
            "role": "system", "content": self._sys_template()
        }]
        self.all_chat_history = [{
            "role": "system", "content": self._sys_template()
        }]
        self.history_limit = history_limit

    @staticmethod
    def _sys_template():
        # return """You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        # Your role is to assist users by answering questions and providing detailed information about Lotus's \
        # brand promotion and its famous historical models. You answers questions \
        # based on snippets of text provided in context. Answer should being as concise as possible."""
        return """You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        Your role is to assist users by answering questions and providing detailed information about Lotus's brand promotion and its famous historical models. \
        You will also receive background information from an internal human assistant, but this information is only for your understanding to better assist the user. \
        Do not include [Internal Assistant] in your responses. \
        Answer the user's questions naturally like human, do not include bullet point directly, incorporating any useful details from the internal assistant's input without explicitly mentioning them. \
        Be as concise as possible in your responses and provide helpful, relevant information to the user.
        """
        
    @staticmethod
    def _qa_template(question, context):
        if context == "":
            return f"""You can respond to the user input based on the following Retrieved Context.
            Retrieved Context: \n{context}\n
            User input: {question}"""
        else:
            return f"""{question}"""

    def if_query_rag(self, question, max_retry=3):
        prompt_template = """
        You are a smart assistant designed to categorize questions. You need to determine whether the user's question is a general daily question or a specific question that requires information from a specific dataset about car statistics.

        The dataset includes detailed historical and technical data about various car models such as:
        - Model names and formulas (e.g., Mark I, Trials Car, Mark II, etc.)
        - Years of production
        - Number of examples built
        - Engine types and specifications (e.g., Austin Seven two-bearing side-valve, Ford 10 side-valve, etc.)
        - Dimensions (length, width, height)
        - Wheelbase
        - Weight

        Here are some example questions related to the dataset:
        - "What engine was used in the Mark I car?"
        - "How many Mark II cars were built?"
        - "Can you provide the specifications for the Mark VI?"
        - "What were the production years for the Mark VIII?"
        - "Please tell me something about lotus?"

        Any question that involves details about car models or mention about the keywords such as lotus, their specifications, history, or technical data should be categorized as requiring the specific dataset (Answer: YES).

        General daily questions might include:
        - "What's the weather like today?"
        - "How do I make a cup of coffee?"
        - "What's the capital of France?"
        - "What time is it?"

        For such questions, the answer should be categorized as not requiring the specific dataset (Answer: NO).

        In addition, if the question requires information from the specific dataset (Answer: YES), please check if the question refers to a specific time period or date. If the user explicitly mentions a time frame, return the most relevant date in the format 'YYYY-MM-DD'. Even if multiple time frames are mentioned, return only the most significant one. If no time is mentioned, do not return any date.

        Reply with "YES" if it requires the dataset, or "NO" if it does not. If you determine that the dataset is required and a time period is mentioned, reply with "YES" followed by the date in the format 'YYYY-MM-DD'. For example, "YES 1965-03-10". If there is no time mentioned, just reply with "YES".

        Remember to only reply with "YES", "YES YYYY-MM-DD", or "NO" without any additional explanation.

        Question: {question}

        """
        tpl = ChatPromptTemplate.from_template(prompt_template)
        chain = tpl | self.llm
        result = "need rag"
        use_time = "no time"
        for i in range(max_retry):
            response = chain.invoke({"question": question})
            if response.content.startswith("YES"):
                if len(response.content) == 3:
                    # 如果只返回了"YES"，不包含日期
                    use_time = "no time"
                elif response.content.startswith("YES ") and len(response.content) > 3:
                    # 如果返回了"YES YYYY-MM-DD"
                    use_time = response.content.split(" ")[1]  # 获取时间信息
            elif response.content == "NO":
                result = "no rag"
                break
        return result, use_time
    

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

    #     return response
    def chat_internal(self, user_input, rag_context='', stream=False, internal_input=None, interrupt_index=None):
        # Handle modification of the previous assistant message if the interrupt index is provided
        if interrupt_index is not None:
            self.modify_previous_assistant_message(interrupt_index)

        # Prepend the internal assistant input to the user message if provided
        if internal_input:
            # Prepend the internal input to the user's message
            user_input = f"[Internal Assistant Information]: {internal_input}\n\nUser Input: {user_input}"

        # Now create the user message and append it to the chat history
        user_message = {"role": "user", "content": self._qa_template(user_input, rag_context)}
        self.chat_history.append(user_message)
        self.all_chat_history.append(user_message)

        data = {
            "model": self.model_name,
            "messages": self.chat_history,
            "stream": stream
        }

        response = requests.post(f"{self.base_url}/api/chat", json=data, stream=True)
        response.raise_for_status()

        return response


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
        assistant_message = {"role": "assistant", "content": response}
        self.chat_history.append(assistant_message)
        self.all_chat_history.append(assistant_message)
        self._trim_chat_history()

    def _trim_chat_history(self):
        # Keep the system message and the last `self.history_limit` user and assistant messages
        non_system_messages = [msg for msg in self.chat_history if msg['role'] != 'system']
        if len(non_system_messages) > self.history_limit:
            self.chat_history = [self.chat_history[0]] + non_system_messages[-self.history_limit:]

    def get_chat_history(self):
        return self.chat_history

    def get_all_chat_history(self):
        return self.all_chat_history

    def clear_chat_history(self):
        self.chat_history = [self.all_chat_history[0]]
