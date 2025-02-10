import logging
logger = logging.getLogger(__name__)

import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List, Dict, Any, Optional

class FaissRetriever:
    """Faiss retriever compatible with LangChain that supports metadata filtering."""
    
    def __init__(self, embeddings, embedding_fn: HuggingFaceEmbeddings):
        super().__init__()
        self.embeddings = embedding_fn
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]

        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(dimension)
        self.index = faiss.index_cpu_to_gpu(res, 0, index)

        x = embeddings.astype('float32')
        faiss.normalize_L2(x)

        self.index.add(x)
        
        logger.info(f"Building FAISS index with {len(embeddings)} vectors of dimension {dimension}")
        logger.debug(f"embeddings shape: {x.shape}")
        # logger.debug(f"first 10 id2uuid: {list(self.id2uuid.items())[:10]}")

    def invoke(
            self,
            querys: list[str],
            k: int
        ):
        query_vec_list = [self.embeddings.embed_query(q) for q in querys]
        query_vector = np.array(query_vec_list).astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        return indices, distances



if "__name__" == "__main__":
    # Example usage:
    retriever = FaissRetriever('/path/to/chroma_db', 128)
    query_vector = [0.1, 0.2, ...]  # Example query vector
    top_k_results = retriever.retrieve(query_vector, top_k=5)
    print(top_k_results)