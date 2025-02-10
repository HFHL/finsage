import logging
import torch
logger = logging.getLogger(__name__)

from typing import Dict, List, Optional, Set, Union, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .bm25Retriever import BM25Retriever
from .faissRetriever import FaissRetriever

class EnsembleRetriever:
    """Base class for retriever wrappers that handle document content retrieval"""
    
    def __init__(self, bm25_dir: str, chroma: Chroma, k: int, embeddings: HuggingFaceEmbeddings):
        super().__init__()
        self.embeddings = embeddings
        self.k = k
        self.chroma = chroma
        docs = chroma.get(include=["metadatas", "embeddings"])

        self.bm25_retriever = BM25Retriever(bm25_dir)
        self.faiss_retriever = FaissRetriever(docs['embeddings'], embeddings)
        
        # save all metadata except title_summary
        self.chunk_metadata = docs['metadatas']
        
        # self.chunk_metadata = [{**metadata, 'chunk_num': 1} for metadata in docs['metadatas']]
        self.docid2idx = {doc['doc_id']: idx for idx, doc in enumerate(self.chunk_metadata)}
        self.num_chunk = len(docs['metadatas'])

        title_summaries = set()

        # get all unique title_summary and embed into faiss index
        for metadata in docs['metadatas']:
            title_summary = metadata.get('title_summary', '')
            if title_summary != '':
                title_summaries.add(title_summary)

        self.title_summaries = list(title_summaries)
        logger.info(f"Building title summary FAISS index with {len(docs['metadatas'])} vectors")
        title_summary_embeddings = embeddings.embed_documents(title_summaries)
        self.title_summary_faiss_retriever = FaissRetriever(title_summary_embeddings, embeddings)
        logger.info("title summary FAISS index built")
        
    
    def invoke(
        self,
        input: str,
        hyde_chunks: list[str],
    ) -> List[Dict]:
        """Get documents with their content"""

        seen_ids = set()
        chunk_list = []
        bundle_cnt = 0

        inputs = [input] + hyde_chunks
        faiss_ids_list, faiss_scores_list = self.faiss_retriever.invoke(inputs, 2048)
        for inp, faiss_ids, faiss_scores in zip(inputs, faiss_ids_list, faiss_scores_list):
            effective_ids = {idx: score for idx, score in zip(faiss_ids, faiss_scores)}
            # augment retrieved content with precious and next chunk
            top_k_ids, top_k_scores = faiss_ids[:self.k], faiss_scores[:self.k]
            logger.info(f"Input: {inp}")
            logger.info(f"Top {self.k} FAISS results:")
            for idx, score in zip(top_k_ids, top_k_scores):
                if idx in seen_ids:
                    continue
                seen_ids.add(idx)
                ids = [idx]
                doc_metadata = self.chunk_metadata[idx]
                # gather bundle if bundle_id is not null
                if doc_metadata.get('bundle_id', None) != None:
                    bundle_id = doc_metadata['bundle_id']
                    # find corresponding bundle_id from self.chunk_metadata
                    bundle_ids = [idx for idx, metadata in enumerate(self.chunk_metadata) if metadata.get('bundle_id', None) == bundle_id]
                    ids = bundle_ids
                    seen_ids.update(bundle_ids)

                # expand chunk if score is high
                if score > 0.72:
                    prev_doc_id = doc_metadata['prev_chunk_id']
                    next_doc_id = doc_metadata['next_chunk_id']
                    while len(ids) < 4:
                        flag = False
                        if prev_doc_id != "" and self.docid2idx.get(prev_doc_id, -1) != -1:
                            prev_id = self.docid2idx[prev_doc_id]
                            if effective_ids.get(prev_id, 0) > 0.66 and prev_id not in seen_ids:
                                flag = True
                                # doc_metadata['chunk_num'] += 1
                                seen_ids.add(prev_id)
                                ids.insert(0, prev_id)
                                prev_doc_id = self.chunk_metadata[prev_id]['prev_chunk_id']

                        if next_doc_id != "" and self.docid2idx.get(next_doc_id, -1) != -1:
                            next_id = self.docid2idx[next_doc_id]
                            if effective_ids.get(next_id, 0) > 0.66 and next_id not in seen_ids:
                                flag = True
                                # doc_metadata['chunk_num'] += 1
                                seen_ids.add(next_id)
                                ids.append(next_id)
                                next_doc_id = self.chunk_metadata[next_id]['next_chunk_id']
                        if not flag:
                            break

                doc_ids = [self.chunk_metadata[idx]['doc_id'] for idx in ids]
                    
                docs_dict = self.chroma.get(ids=doc_ids, include=['documents', 'metadatas'])

                # candidate chunks bring the whole bundle
                logger.info(f"Bundle {bundle_cnt}")
                for idx in range(len(docs_dict['documents'])):
                    logger.info(f"{len(chunk_list)} chunk score: {effective_ids.get(self.docid2idx[docs_dict['metadatas'][idx]['doc_id']], 0)}")
                    logger.info(f"{len(chunk_list)} chunk doc_id: {docs_dict['metadatas'][idx].get('doc_id', '')}")
                    logger.info(f"{len(chunk_list)} chunk content: {docs_dict['documents'][idx]}")

                    metadata_copy = docs_dict['metadatas'][idx].copy()
                    metadata_copy.pop('title_summary', None)                   
                    chunk_list.append(

                        {   "retriever": "faiss",
                            "page_content": docs_dict['documents'][idx],
                            "metadata": metadata_copy,
                            "bundle_id": bundle_cnt
                        }
                    )
                    
                bundle_cnt += 1

        title_summary_ids, title_summary_scores = self.title_summary_faiss_retriever.invoke([input], 5)
        title_summary_ids, title_summary_scores = title_summary_ids[0], title_summary_scores[0]
        logger.info(f"Top {self.k} Title Summary FAISS results:")
        for title_idx, score in zip(title_summary_ids, title_summary_scores):
            title_summary = self.title_summaries[title_idx]
            # find corresponding chunk idx from self.chunk_metadata
            chunk_idxs = [idx for idx, metadata in enumerate(self.chunk_metadata) if metadata.get('title_summary', '') == title_summary]
            logger.info("score: {score} title_summary: {title_summary}".format(score=score, title_summary=title_summary.replace('\n', ' ')))
            for idx in chunk_idxs:
                if idx in seen_ids:
                    continue
                seen_ids.add(idx)
                ids = [idx]
                doc_metadata = self.chunk_metadata[idx]
                # gather bundle if bundle_id is not null
                if doc_metadata.get('bundle_id', None) != None:
                    bundle_id = doc_metadata['bundle_id']
                    # find corresponding bundle_id from self.chunk_metadata
                    bundle_ids = [idx for idx, metadata in enumerate(self.chunk_metadata) if metadata.get('bundle_id', None) == bundle_id]
                    ids = bundle_ids
                    seen_ids.update(bundle_ids)

                # get content of ids
                doc_ids = [self.chunk_metadata[idx]['doc_id'] for idx in ids]
                docs_dict = self.chroma.get(ids=doc_ids, include=['documents', 'metadatas'])
                doc_content = "\n".join(docs_dict['documents'])
                title_summary = doc_metadata.get('title_summary', '').replace('\n', ' ')

                # candidate chunks bring the whole bundle
                logger.info(f"Bundle {bundle_cnt}")
                for idx in range(len(docs_dict['documents'])):
                    logger.info(f"{len(chunk_list)} chunk doc_id: {docs_dict['metadatas'][idx].get('doc_id', '')}")
                    logger.info(f"{len(chunk_list)} chunk content: {docs_dict['documents'][idx]}")
                    metadata_copy = docs_dict['metadatas'][idx].copy()
                    metadata_copy.pop('title_summary', None)
                    chunk_list.append(
                        {
                            "retriever": "title_summary_faiss",
                            "page_content": docs_dict['documents'][idx],
                            "metadata": metadata_copy,
                            "bundle_id": bundle_cnt
                        }
                    )

                bundle_cnt += 1

        bm25_ids, bm25_scores = self.bm25_retriever.invoke(input, self.num_chunk)
        top_k_ids, top_k_scores = bm25_ids[:self.k], bm25_scores[:self.k]
        
        logger.info(f"Top {self.k} BM25 results:")
        for idx, score in zip(top_k_ids, top_k_scores):
            if idx in seen_ids:
                continue
            seen_ids.add(idx)
            ids = [idx]
            doc_metadata = self.chunk_metadata[idx]
            # gather bundle if bundle_id is not null
            if doc_metadata.get('bundle_id', None) != None:
                bundle_id = doc_metadata['bundle_id']
                # find corresponding bundle_id from self.chunk_metadata
                bundle_ids = [idx for idx, metadata in enumerate(self.chunk_metadata) if metadata.get('bundle_id', None) == bundle_id]
                ids = bundle_ids
                seen_ids.update(bundle_ids)

            # get content of ids
            doc_ids = [self.chunk_metadata[idx]['doc_id'] for idx in ids]
            docs_dict = self.chroma.get(ids=doc_ids, include=['documents', 'metadatas'])

            # candidate chunks bring the whole bundle
            logger.info(f"Bundle {bundle_cnt} score: {score}")
            for idx in range(len(docs_dict['documents'])):
                logger.info(f"{len(chunk_list)} chunk doc_id: {docs_dict['metadatas'][idx].get('doc_id', '')}")
                logger.info(f"{len(chunk_list)} chunk content: {docs_dict['documents'][idx]}")
                metadata_copy = docs_dict['metadatas'][idx].copy()
                metadata_copy.pop('title_summary', None)  
                chunk_list.append(
                    {
                        "retriever": "BM25",
                        "page_content": docs_dict['documents'][idx],
                        "metadata": metadata_copy,
                        "bundle_id": bundle_cnt                       
                    }
                )

            bundle_cnt += 1
            
        return chunk_list

    def compute_similarity(self, chunks: List[str], selected_indices: List[int], candidate_index: int) -> List[float]:
        """
        计算 candidate_index 对应 chunk 和 selected_indices 对应 chunks 的相似度（GPU 加速）。
        
        参数:
            chunks (List[str]): 文档块的字符串列表。
            selected_indices (List[int]): 选定的索引列表。
            candidate_index (int): 候选索引。
            
        返回:
            List[float]: candidate_index 对应 chunk 和 selected_indices 对应 chunks 的相似度列表。
        """
        # 将字符串转化为嵌入向量
        embeddings = torch.stack([torch.tensor(self.embeddings.embed_query(chunk), device='cuda') for chunk in chunks])
        
        # 提取 candidate_index 对应的嵌入向量
        candidate_embedding = embeddings[candidate_index].unsqueeze(0)  # 添加 batch 维度
        
        # 提取 selected_indices 对应的嵌入向量
        selected_embeddings = embeddings[selected_indices]

        # 归一化嵌入向量
        candidate_embedding = torch.nn.functional.normalize(candidate_embedding, dim=-1)
        selected_embeddings = torch.nn.functional.normalize(selected_embeddings, dim=-1)
        
        # 计算余弦相似度 (使用点积)
        similarity = torch.matmul(selected_embeddings, candidate_embedding.T).squeeze(-1)
        
        return similarity
    
    def compute_similarity_mtx(self, chunks: List[str]) -> torch.Tensor:
        """
        计算 chunks 两两之间的相似度矩阵（GPU 加速）。
        
        参数:
            chunks (List[str]): 文档块的字符串列表。
            
        返回:
            torch.Tensor: chunks 两两之间的相似度矩阵。
        """
        embeddings = torch.stack([torch.tensor(self.embeddings.embed_query(chunk), device='cuda') for chunk in chunks])
        
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        similarity_mtx = torch.matmul(embeddings, embeddings.T)
        
        return similarity_mtx
