import os
import yaml
import logging
logger = logging.getLogger(__name__)

from datetime import datetime
from typing import Dict, List, Optional, Set, Union, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .ensembleRetriever import EnsembleRetriever
import GPUtil

class RAGManager:
    """Singleton class for managing RAG collections"""
    _collections: Dict[str, Chroma] = {}
    _retrievers: List[EnsembleRetriever] = []

    _instance = None
    _config = None

    def __new__(cls, config: Dict = None, collections: Dict[str, int] = None):
        if cls._instance is None:
            if config is None:
                logger.error("No config provided")
                raise ValueError("No config provided for RAGManager")
            cls._instance = super(RAGManager, cls).__new__(cls)
            cls._instance._initialize(config, collections)
        return cls._instance

    def __init__(self, config: Dict = None, collections: Dict[str, int] = None):
        pass

    def _initialize(self, config: Dict, collections: Dict[str, int]):
        self._config = config
        self.persist_directory = os.path.join(config['persist_directory'], "chroma")
        self.embeddings_model_name = config['embeddings_model_name']
        self.batch_size = 5
        try:
            logger.info("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
            # st_model = self.embeddings.client
            # print("Model name from config:", st_model._modules['0'].auto_model.config.model_type)
            # print("Model path:", st_model._modules['0'].auto_model.config._name_or_path)
            # print("Model files location:", st_model._modules['0'].auto_model.config.name_or_path)
            # print("Full model config:", st_model._modules['0'].auto_model.config)
            # print("Model files:", st_model._modules['0'].auto_model._parameters)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

        if collections is not None:
            for collection, top_k in collections.items():
                if top_k <= 0:
                    continue
                self.create_collection(collection)
                self._retrievers.append(self.create_retriever(top_k, collection, retriever_type="ensemble"))

        
    def create_collection(self, collection_name: str):
        """Create a new collection with all supported retrievers"""
        if collection_name not in self._collections:
            # Initialize Chroma
            chroma = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                relevance_score_fn="l2" # l2, ip, cosine
            )
            self._collections[collection_name] = chroma

    def get_collection_documents(self, collection_name: str, doc_ids: Optional[List[str]] = None) -> List[Document]:
        """Get documents from a collection by document IDs. User should not assume that the order of the returned documents matches the order of the input IDs."""
        if doc_ids is None:
            chroma_docs = self._collections[collection_name].get()
        else:
            chroma_docs = self._collections[collection_name].get(ids=doc_ids)

        documents = [
            Document(
                page_content=page_content,
                metadata=metadata
            )
            for page_content, metadata in zip(chroma_docs['documents'], chroma_docs['metadatas'])
        ]
        return documents

    def create_retriever(self, k: int, collection_name: str, retriever_type: str = "chroma"):
        """Create a specific retriever for a collection"""
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        bm25_dir = os.path.join(self._config['persist_directory'], "bm25_index", collection_name)

        retriver = EnsembleRetriever(bm25_dir, self._collections[collection_name], k, self.embeddings)
            
        return retriver


# Usage example
def main():
    config_path = "../../config/config_test.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    questions = [
        "Are there any new releases in 2023?",
        "Can you tell me how Lotus's approach to vehicle design evolved between 2000 and 2020?",
        "What are the unique technical features that make Lotus stand out in racing?" ,
        "Can you explain the lightweight design philosophy of Lotus?" ,
        "Which Lotus models are best known for their driving performance on the track?" ,
    ]
    
    rag = RAGManager(config)
    log_gpu_usage('RAGManager init')
    rag.create_collection("lotus")
    log_gpu_usage('RAGManager create collection')
    retriever = rag.create_retriever(5, "lotus", "ensemble")
    log_gpu_usage('RAGManager get retriever')

    for q in questions:
        documents = retriever.invoke(q)
        log_gpu_usage('RAGManager invoke retriever')
        print(f"Question: {q}")
        for i, doc in enumerate(documents):
            print(f"{i}: {doc}")
        print("")
        

def log_gpu_usage(event_name):
    gpus = GPUtil.getGPUs()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gpu_log_file = "gpu_usage.log"
    for gpu in gpus:
        gpu_info = (
            f"Timestamp: {timestamp}, Event: {event_name}, "
            f"GPU ID: {gpu.id}, GPU Name: {gpu.name}, "
            f"Memory Used: {gpu.memoryUsed} MB, Memory Total: {gpu.memoryTotal} MB"
        )
        # 将信息追加到日志文件
        with open(gpu_log_file, 'a') as f:
            f.write(gpu_info + '\n')

if __name__ == "__main__":
    main()
