<div style="text-align: center;">
<img src="https://pic1.imgdb.cn/item/67b72cb4d0e0a243d4010f0f.png" width="100%" height="auto" />
</div>

# 📚 FinSage: Multi-modal RAG QA System for Financial Documents
---

## 🖊️ Project Overview

FinSage is an intelligent framework specifically designed for the financial sector, addressing compliance analysis challenges in financial document workflows. While enterprises in the financial industry typically rely on Retrieval-Augmented Generation (RAG) systems to handle complex compliance requirements, existing solutions often struggle with data heterogeneity (e.g., text, tables, charts) and evolving regulatory standards, impacting information extraction accuracy. To address these challenges, FinSage introduces three innovative technologies:

1. Multi-modal Preprocessing Pipeline: Unifies processing of various data formats and generates metadata summaries for data chunks, enabling effective integration and analysis of heterogeneous data.
2. Multi-path Sparse-Dense Retrieval System: Combines query expansion and metadata-aware semantic search (HyDE) to achieve precise retrieval from large-scale document repositories.
3. Domain-specific Reranking Module: Fine-tuned through Direct Preference Optimization (DPO) to prioritize compliance-related key information, ensuring outputs align with financial sector regulations.

Experimental results show that FinSage achieves a recall rate of 92.51% on the FinanceBench dataset, improving accuracy by 24.06% compared to the best baseline method. Currently, FinSage has been successfully deployed as a financial QA agent, serving over 1,200 users in online meetings. The system is now open-source and available for public use.

<div style="text-align: center;">
<img src="https://pic1.imgdb.cn/item/67b72cb4d0e0a243d4010f10.png" width="100%" height="auto" />
</div>

## 💻 Deployment Guide

### Environment Requirements
- Python Version: 3.10.14
- Install Dependencies: `pip install -r environment.txt`

### Data Processing
1. Extract PDF content using MinerU (Reference: https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html)
```bash
magic-pdf -p {pdf_path} -o ./data/chunk -m auto
```
where `pdf_path` is the path to your PDF file.

2. Navigate to `./file2chunk`
   - Modify the `root_folder` variable in `main_pipeline.py`'s `main` function to point to the `/auto` path in MinerU's output directory
   - Specify the output path
   - Run the processing pipeline and place the generated JSON files in `./data/chunk` directory

### System Configuration
1. Modify `./config/config_vllm.yaml`
   - Set `persist_directory` for ChromaDB persistence

2. Data Loading
   - Navigate to `./script`
   - Update `collection0_dir` variable to point to your JSON file storage path
   - Execute data loading:
```bash
python load_data.py
```

### Model Deployment
1. Download models (See `./models/models.md` for details)

2. Load model using VLLM:
```bash
nohup vllm serve Qwen/Qwen2___5-72B-Instruct-AWQ --max-model-len 5120 --gpu_memory_utilization 0.65 --enforce-eager --swap-space 36 --disable-log-stats --uvicorn-log-level warning > vllm.log 2>&1 &
```

### Launch Service
```bash
cd ./src
python app2.py
```

Access the web chat interface at `localhost:6005/test_api_chat`.