<div style="text-align: center;">
<img src="https://pic1.imgdb.cn/item/67b72cb4d0e0a243d4010f0f.png" width="100%" height="auto" />
</div>

# 📚 FinSage：面向金融文档的多模态 RAG 问答系统
---

## 🖊️ 项目简介

FinSage 是一个专为金融领域设计的智能框架，致力于解决金融文档工作流中的合规性分析问题。在金融行业，企业通常依赖检索增强生成（RAG）系统来应对复杂的合规要求。然而，现有解决方案往往难以有效处理数据的异质性（如文本、表格、图表等）以及动态变化的法规标准，这显著影响了信息提取的准确性。为此，FinSage 引入了三项创新技术：

1. 多模态预处理管道：统一处理多种数据格式，生成数据块的元数据摘要，有效整合和分析异构数据。
2. 多路径稀疏密集检索系统：结合查询扩展和元数据感知语义搜索（HyDE），实现大规模文档库中的精确检索。
3. 领域专用重排序模块：通过直接偏好优化（DPO）微调，优先提取合规相关的关键信息，确保输出结果符合金融领域规范。

实验结果显示，FinSage 在 FinanceBench 数据集上取得了 92.51% 的召回率，较最优基线方法提升了 24.06% 的准确性。目前，FinSage 已成功部署为金融问答代理，在线上会议中服务超过 1200 名用户。我们已开源此系统，欢迎访问使用。

<div style="text-align: center;">
<img src="https://pic1.imgdb.cn/item/67b72cb4d0e0a243d4010f10.png" width="100%" height="auto" />
</div>

## 💻 部署指南

### 环境要求
- Python 版本：3.10.14
- 依赖安装：`pip install -r environment.txt`

### 数据处理
1. 使用 MinerU 提取 PDF 内容（参考：https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html）
```bash
magic-pdf -p {pdf_path} -o ./data/chunk -m auto
```
其中 `pdf_path` 为待处理 PDF 文件的路径。

2. 进入 `./file2chunk` 目录
   - 修改 `main_pipeline.py` 中 `main` 函数的 `root_folder` 变量，指向 MinerU 输出目录中的 `/auto` 路径
   - 指定输出路径
   - 运行处理流程，生成的 JSON 文件需放置于 `./data/chunk` 目录

### 系统配置
1. 修改 `./config/config_vllm.yaml`
   - 设置 `persist_directory` 为 ChromaDB 持久化路径

2. 数据加载
   - 进入 `./script` 目录
   - 修改 `collection0_dir` 变量为 JSON 文件存储路径
   - 执行数据加载：
```bash
python load_data.py
```

### 模型部署
1. 下载模型（详见 `./models/models.md`）

2. 使用 VLLM 加载模型：
```bash
nohup vllm serve Qwen/Qwen2___5-72B-Instruct-AWQ --max-model-len 5120 --gpu_memory_utilization 0.65 --enforce-eager --swap-space 36 --disable-log-stats --uvicorn-log-level warning > vllm.log 2>&1 &
```

### 启动服务
```bash
cd ./src
python app2.py
```

访问 `localhost:6005/test_api_chat` 即可使用网页对话界面.