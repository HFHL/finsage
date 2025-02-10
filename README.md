RAG_Agent

Directory Structure

config/
	•	Contains configuration files, such as config_vllm.yaml
	•	Includes configuration information such as database paths, model names, API URLs, etc.

file2chunk/
	•	Core scripts for file handling and chunking
	•	Main functionalities include:
	•	File content analysis and word count (content_word_count.py)
	•	Slide content extraction (extract_slide.py)
	•	Main workflow for file processing (main_pipeline.py)

script/
	•	Directory for auxiliary scripts
	•	Contains:
	•	Output editing tool (editOutput.py)
	•	Data loading script (load_data.py)

src/
	•	Project source code directory
	•	Mainly includes:
	•	Web application services (app.py, app2.py)
	•	GPU monitoring and logging tools (gpu_log.py, gpu_monitor.py)
	•	Front-end templates (templates/)
	•	Core tools for the RAG system
	•	Test code (test/)
	•	API testing
	•	Evaluation scripts
	•	Performance analysis tools

Main Features
	1.	Document Processing
	•	Supports content extraction from PDFs and other formats
	•	Text chunking and deduplication
	•	Coreference resolution
	•	Intelligent summary generation
	2.	RAG System
	•	Retrieval based on vector databases
	•	Text similarity analysis
	•	Context understanding and coreference resolution
	3.	Web Services
	•	RESTful API interfaces
	•	Real-time conversation capabilities
	•	GPU resource monitoring
	4.	Testing and Evaluation
	•	Performance evaluation tools
	•	Data quality analysis
	•	Automated testing scripts