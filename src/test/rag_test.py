import os
import sys
import yaml
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chromaManager import ChromaManager
from utils.ollamaManager import OllamaManager


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def test_retrieval_acc(config, collection_name, test_directory='../Data/test_data', k=10):
    manager = ChromaManager(config, collection_name)
    manager.create_collection()

    test_queries = []
    for filename in os.listdir(test_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(test_directory, filename)
            with open(file_path, 'r') as file:
                test_queries.extend(json.load(file))

    accuracy = manager.evaluate_retrieval(test_queries, k, "ensemble", False)
    # accuracy = manager.evaluate_retrieval(test_queries, k, "chroma", False)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def test_if_query_rag(config, collection_name, test_directory):
    chroma_manager = ChromaManager(config, collection_name)
    chroma_manager.create_collection()
    db_ret = chroma_manager.get_db_as_ret(search_kwargs={"k": 10})
    ollama_manager = OllamaManager(config, db_ret)

    test_queries = []
    for filename in os.listdir(test_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(test_directory, filename)
            with open(file_path, 'r') as file:
                test_queries.extend(json.load(file))

    num = 0
    for test_query in tqdm(test_queries):
        question = test_query['question']
        true_label = test_query['label']
        test_label = ollama_manager.if_query_rag(question)
        if true_label == test_label:
            num += 1
        else:
            print(f"Question: {question}\ntrue_label: {true_label}, test_label: {test_label}")

    print(f"Total test query: {len(test_queries)}, "
          f"Correct Number: {num}, "
          f"Accuracy: {num / len(test_queries) * 100:.2f}%")

def ask_question_and_print_source(config, collection_name, question, k=10):
    chroma_manager = ChromaManager(config, collection_name)
    chroma_manager.create_collection()
    retriever = chroma_manager.get_retriever(k=k, retriever_type="chroma")
    ollama_manager = OllamaManager(config, retriever)
    
    response = ollama_manager.if_query_rag(question)
    
    chroma_docs = retriever.invoke(question)
    sources = []
    for doc in chroma_docs:
        content = doc.page_content
        metadata = doc.metadata
        sources.append({
            "content": content,
            "filename": metadata.get("filename", "Unknown file"),
            "page_number": metadata.get("page_number", "Unknown page")
        })

    print(f"Question: {question}")
    print(f"Response: {response}")
    print("Sources:")
    for source in sources:
        print(f"- Filename: {source['filename']}, Page Number: {source['page_number']}")
        #print(f"  Content: {source['content']}")
        #print(f"- {source}")

if __name__ == '__main__':
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    
    question = "Are there any new releases in 2023?"
    ask_question_and_print_source(config, 'lotus_car_stats', question)
    
    #test_retrieval_acc(config, 'lotus_car_stats', '/root/autodl-tmp/RAG_Agent/data/test_data/test_rag', k=10)
    # test_if_query_rag(config, 'lotus', '/root/autodl-tmp/RAG_Agent/Data/test_need_rag')
