# RAG_Agent

## Directory Structure

### config/
- **Purpose:** Contains configuration files.
- **Details:**
  - Example file: `config_vllm.yaml`
  - Stores configuration information such as:
    - Database paths
    - Model names
    - API URLs
    - Other relevant settings

### file2chunk/
- **Purpose:** Core scripts for file handling and chunking.
- **Main Functionalities:**
  - **File Content Analysis and Word Count:**  
    - Script: `content_word_count.py`
  - **Slide Content Extraction:**  
    - Script: `extract_slide.py`
  - **Main Workflow for File Processing:**  
    - Script: `main_pipeline.py`

### script/
- **Purpose:** Directory for auxiliary scripts.
- **Contents:**
  - **Output Editing Tool:**  
    - Script: `editOutput.py`
  - **Data Loading Script:**  
    - Script: `load_data.py`

### src/
- **Purpose:** Project source code directory.
- **Contents:**
  - **Web Application Services:**  
    - Scripts: `app.py`, `app2.py`
  - **GPU Monitoring and Logging Tools:**  
    - Scripts: `gpu_log.py`, `gpu_monitor.py`
  - **Front-end Templates:**  
    - Directory: `templates/`
  - **Core Tools for the RAG System**
    - `apiOllamaManager.py`: Manages chat interactions with Ollama API, including history tracking and RAG integration
    - `bm25Retriever.py`: BM25 retriever implementation for document search
    - `chatService.py`: Core service handling chat interactions and RAG orchestration
    - `ensembleRetriever.py`: Combines multiple retrievers (BM25, FAISS) for improved document retrieval
    - `faissRetriever.py`: FAISS-based vector retriever for semantic search
    - `ragManager.py`: Manages RAG collections and retrievers
    - `vllmChatService.py`: vLLM-specific chat service implementation
    - `vllmManager.py`: Manages vLLM model interactions and chat history
  - **Test Code:** Located in the `test/` directory, including:
    - API testing scripts
    - Evaluation scripts
    - Performance analysis tools

## Main Features

1. **Document Processing**
   - Supports content extraction from PDFs and other formats.
   - Implements text chunking and deduplication.
   - Provides coreference resolution.
   - Generates intelligent summaries.

2. **RAG System**
   - Utilizes vector databases for retrieval.
   - Performs text similarity analysis.
   - Supports context understanding and coreference resolution.

3. **Web Services**
   - Offers RESTful API interfaces.
   - Enables real-time conversation capabilities.
   - Monitors GPU resources.

4. **Testing and Evaluation**
   - Includes performance evaluation tools.
   - Provides data quality analysis.
   - Contains automated testing scripts.