import psutil
import logging
import yaml
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
import uuid
import time
import concurrent.futures #added
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class RAGToTNode:
    def __init__(self, question, documents, summary, is_sufficient):
        self.id = str(uuid.uuid4())
        self.question = question
        self.documents = documents
        self.summary = summary
        self.is_sufficient = is_sufficient
        self.children = []
        self.combined_summary = None
class RAGToT:
    def __init__(self, config, collection_name, batch_size=5,max_workers=5):
        self.file_path = config['file_path']
        self.persist_directory = config['persist_directory']
        self.embeddings_model_name = config['embeddings_model_name']
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.store = {}
        mem = psutil.virtual_memory()
        available_memory_gb = mem.available / (1024 ** 3)
        logging.info(f"Initializing ChromaManager - Available memory: {available_memory_gb:.2f} GB")
        self.current_tree = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


    def load_model(self, model_name="llama3:8b"):
        try:
            logging.info("Loading embedding model...")

            mem_before = psutil.virtual_memory()
            available_memory_before_gb = mem_before.available / (1024 ** 3)
            logging.info(f"Memory before loading model: {available_memory_before_gb:.2f} GB")

            self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
            self.llm = ChatOllama(model=model_name)
            self.llm2 = ChatOllama(model="gemma2:27b")
            print(f"run {model_name}")

            mem_after = psutil.virtual_memory()
            available_memory_after_gb = mem_after.available / (1024 ** 3)
            logging.info(f"Memory after loading model: {available_memory_after_gb:.2f} GB")

            memory_used_gb = available_memory_before_gb - available_memory_after_gb
            logging.info(f"Memory used by model: {memory_used_gb:.2f} GB")

            self.chroma_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
        
    def retrieve_documents(self, question, num_docs=3):
        docs = self.chroma_db.similarity_search(question, k=num_docs)
        return [{
            'content': doc.page_content,
            'car_stats': doc.metadata.get('car_stats', 'No car stats available')
        } for doc in docs]

    def extract_and_summarize(self, question, documents):
        # Separate documents and car stats
        documents_text = "\n\n".join(doc['content'] for doc in documents)
        car_stats_text = "\n".join(doc['car_stats'] for doc in documents)

        prompt = f"""
        Question: {question}
        
        Documents:
        {documents_text}
        
        Car stats Related to retrieved Documents:
        {car_stats_text}
        
        Extract and summarize the key information from these documents and car stats that could answer the question. 
        Provide a concise summary in around seven sentence.
        """
        response = self.llm.invoke(prompt)
        return response.content
    
    def check_sufficiency(self, question, summary):
        prompt = f"""
        Question: {question}
        
        Summarized Information:
        {summary}
        
        Is this information and your internal knowledge sufficient to fully answer the question? 
        Respond with 'YES' if sufficient, or 'NO' if more information is needed.
        """
        response = self.llm.invoke(prompt)
        #print(response)
        return response.content.strip().upper() == 'YES'
    
    def generate_questions(self, original_question, summary, num_questions=3):
        prompt = f"""
        Original Question: {original_question}
        
        Current Information:
        {summary}
        
        Generate {num_questions} follow-up questions to gather more relevant information. 
        If no more questions can be asked, rephrase the original question in a different way.
        The genrated Format should be
        1. Question1 
        2. Question2
        etc.
        """
        response = self.llm.invoke(prompt)
        return [q.strip() for q in response.content.split('\n') if q.strip()]
    
    def process_question(self, question, documents):
        summary = self.extract_and_summarize(question, documents)
        is_sufficient = self.check_sufficiency(question, summary)
        follow_up_questions = self.generate_questions(question, summary)[1:] if not is_sufficient else []
        return summary, is_sufficient, follow_up_questions
    
    # def rag_tot(self, question, max_depth=3, current_depth=0):
    #     if current_depth >= max_depth:
    #         return self.retrieve_documents(question)

    #     documents = self.retrieve_documents(question)
    #     summary = self.extract_and_summarize(question, documents)

    #     if self.check_sufficiency(question, summary):
    #         return summary

    #     follow_up_questions = self.generate_questions(question, summary)[1:]
    #     print(documents)
    #     print(follow_up_questions)
    #     additional_info = []
    #     for sub_question in follow_up_questions:
    #         sub_info = self.rag_tot(sub_question, max_depth, current_depth + 1)
    #         if isinstance(sub_info, list):
    #             sub_info = "\n\n".join(sub_info)
    #         additional_info.append(sub_info)

    #     combined_info = summary + "\n\nAdditional Information:\n" + "\n".join(additional_info)
    #     return self.extract_and_summarize(question, [combined_info])

    def combine_summaries(self, node):
        prompt = f"""
        Main Question: {node.question}
        
        Main Summary: {node.summary}
        
        Additional Information from Follow-up Questions:
        {self.format_child_summaries(node.children)}
        
        Provide a comprehensive summary that addresses the main question, 
        incorporating both the main summary and the additional information 
        from follow-up questions. Ensure the summary is coherent and directly 
        relevant to the main question. Please keep it concise and around seven setences.
        """
        response = self.llm.invoke(prompt)
        return response.content

    def format_child_summaries(self, children):
        formatted_summaries = []
        for child in children:
            formatted_summaries.append(f"Follow-up Question: {child.question}\nSummary: {child.combined_summary}")
        return "\n\n".join(formatted_summaries)
    
    def rag_tot(self, question, max_depth=3, current_depth=0):
        documents = self.retrieve_documents(question)
        summary, is_sufficient, follow_up_questions = self.process_question(question, documents)
        print(current_depth)
        node = RAGToTNode(question, documents, summary, is_sufficient)

        if current_depth == 0:
            self.current_tree = node

        if is_sufficient or current_depth >= max_depth:
            return node

        futures = [self.executor.submit(self.rag_tot, q, max_depth, current_depth + 1) for q in follow_up_questions]
        for future in concurrent.futures.as_completed(futures):
            child_node = future.result()
            node.children.append(child_node)

        # Combine current summary with child summaries
        node.combined_summary = self.combine_summaries(node)

        return node

    def answer_question(self, question, max_depth=2):
        root_node = self.rag_tot(question, max_depth)
        final_summary = root_node.combined_summary
        
        prompt = f"""
        Question: {question}
        
        Information:
        {final_summary}

        Based on the above information and your internal knowledge, \
        provide a comprehensive answer to the question in around seven sentences.
        """
        # Based on the above information, which includes details from follow-up questions,\
        # If you don't know the answer, just say that you don't know. \
        # Use Seven sentences maximum and keep the answer concise.
        response = self.llm2.invoke(prompt)
        return response.content
    
    def run(self, question, max_depth=3):
        try:
            logging.info(f"Processing question: {question}")
            answer = self.answer_question(question, max_depth)
            logging.info("Answer generated successfully")
            return answer
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            return f"An error occurred: {str(e)}"
        finally:
            pass

    def __del__(self):
        # Ensure the executor is shut down when the instance is deleted
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

    #Added
    def get_tree_data(self):
        if not self.current_tree:
            return None
        return self._node_to_dict(self.current_tree)

    def _node_to_dict(self, node):
        return {
            "id": node.id,
            "question": node.question,
            "is_sufficient": node.is_sufficient,
            "children": [self._node_to_dict(child) for child in node.children]
        }

    def get_node_details(self, node_id):
        return self._find_node(self.current_tree, node_id)

    def _find_node(self, node, node_id):
        if node.id == node_id:
            return {
                "question": node.question,
                "documents": node.documents,
                "summary": node.summary,
                "is_sufficient": node.is_sufficient
            }
        for child in node.children:
            result = self._find_node(child, node_id)
            if result:
                return result
        return None
def main():
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    rag_tot = RAGToT(config, 'lotus',max_workers=7)
    rag_tot.load_model("gemma2:9b")
    result = rag_tot.run("Lotus在F1历史上有哪些值得铭记的成就? 能否分享一些具体的故事或数据?", max_depth=2)
    print("Answer:")
    print(result)
    starttime = time.time()
    result = rag_tot.run("Lotus的哪些技术创新对整个汽车行业产生了深远影响?", max_depth=2)
    endtime = time.time()
    print("Answer:")
    print(f"Time spent to generate result {endtime - starttime}")
    print(result)

if __name__ == '__main__':
    main()