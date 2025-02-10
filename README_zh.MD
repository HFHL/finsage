# RAG_Agent

## 目录结构

### config/
- 存放配置文件，如 config_vllm.yaml
- 包含数据库路径、模型名称、API URL等配置信息

### file2chunk/
- 用于文件处理和分块的核心脚本
- 主要功能包括:
  - 文件内容分析和统计 (content_word_count.py)
  - 幻灯片内容提取 (extract_slide.py)
  - 文件处理主流程 (main_pipeline.py)


### script/
- 辅助脚本目录
- 包含:
  - 输出编辑工具 (editOutput.py)
  - 数据加载脚本 (load_data.py)

### src/
- 项目源代码目录
- 主要包含:
  - Web应用服务 (app.py, app2.py)
  - GPU监控和日志工具 (gpu_log.py, gpu_monitor.py)
  - 前端模板 (templates/)
  - RAG系统核心工具
  - 测试代码 (test/)
    - API测试
    - 评估脚本
    - 性能分析工具

## 主要功能

1. 文档处理
- 支持PDF、的内容提取
- 文本分块和去重
- 指代消解
- 智能摘要生成

2. RAG系统
- 基于向量数据库的检索
- 文本相似度分析
- 上下文理解和指代消解

3. Web服务
- RESTful API接口
- 实时对话功能
- GPU资源监控

4. 测试与评估
- 性能评估工具
- 数据质量分析
- 自动化测试脚本