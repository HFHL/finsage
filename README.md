# RAG_Agent


## Update
* 2024-12-11: 在rerank中加入时间bonus：对与query时间越近的时间点bonus越高, $bonus \in [0,1]$


## 开启服务
1. 启动vllm服务

   1. 进入vllm目录 `cd /root/autodl-tmp/models_vllm/`
   2. 激活conda环境 `conda activate vllm`
   3. `nohup vllm serve Qwen/Qwen2___5-72B-Instruct-AWQ --max-model-len 5120 --gpu_memory_utilization 0.65 --enforce-eager --swap-space 36 --disable-log-stats --uvicorn-log-level warning > vllm.log 2>&1 &`
    * vllm默认启动在`8000`端口
    * 参数
      * `max-model-len: context length (Default 32768)`
      * `gpu-memory_utilization: fraction of GPU memory to be used tor the model executor (Global limit) (Default 0.9)`
      * `enforce-eager: Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility. (Default false)`
      * `swap-space: CPU swap space size (GiB) per GPU. (Default 4)`
    * 其他参数
      * `--kv-cache-dtype fp8_e5m2`
      * `--disable-frontend-multiprocessing`
      * [`vllm server` doc](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-serve)

2. 启动app
   1. 进入`RAG_Agent_vllm/src`
   2. 后台持续运行app2 `nohup python app2.py > app.log 2>&1 &`
3. 开启ngrok反向代理
   
   `ngrok http --url=cat-mighty-factually.ngrok-free.app 6005`
4. 访问网站
   
   `https://cat-mighty-factually.ngrok-free.app/test_api_chat`

## 关闭服务
   1. 关闭app
      1. `lsof -i :6005` 找到app的PID
      2. `kill [pid]`
   2. 关闭vllm
      1. `ps aux | grep vllm` 找到和启动vllm对应的进程
      2. `kill -2 [pid]` 发送SIGINT，完成cleanup

## 查看日志
1. app
   1. app console输出

      `RAG_Agent_vllm/src/app.log` 

   2. logger输出
   
      `RAG_Agent_vllm/src/app_rag.log`

2. vllm

   `/root/autodl-tmp/models_vllm/vllm.log`