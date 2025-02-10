Here is the translated version of your document:

RAG_Agent

Update
	•	2024-12-11: Added a time bonus in rerank: the closer the time point is to the query time, the higher the bonus. ￼

Start the Service
	1.	Start the vllm service
	1.	Enter the vllm directory: cd /root/autodl-tmp/models_vllm/
	2.	Activate the conda environment: conda activate vllm
	3.	Run the following command in the background:

nohup vllm serve Qwen/Qwen2___5-72B-Instruct-AWQ --max-model-len 5120 --gpu_memory_utilization 0.65 --enforce-eager --swap-space 36 --disable-log-stats --uvicorn-log-level warning > vllm.log 2>&1 &

	•	vllm runs on port 8000 by default
	•	Parameters:
	•	max-model-len: Context length (Default 32768)
	•	gpu-memory_utilization: Fraction of GPU memory to be used for the model executor (Global limit) (Default 0.9)
	•	enforce-eager: Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid mode for maximum performance and flexibility. (Default false)
	•	swap-space: CPU swap space size (GiB) per GPU. (Default 4)
	•	Other parameters:
	•	--kv-cache-dtype fp8_e5m2
	•	--disable-frontend-multiprocessing
	•	vllm server documentation

	2.	Start the app
	1.	Navigate to RAG_Agent_vllm/src
	2.	Run the app in the background:

nohup python app2.py > app.log 2>&1 &


	3.	Start ngrok reverse proxy

ngrok http --url=cat-mighty-factually.ngrok-free.app 6005


	4.	Access the website

https://cat-mighty-factually.ngrok-free.app/test_api_chat



Stop the Service
	1.	Stop the app
	1.	Find the app’s PID:

lsof -i :6005


	2.	Kill the process:

kill [pid]


	2.	Stop vllm
	1.	Find the corresponding process for vllm:

ps aux | grep vllm


	2.	Send SIGINT to complete cleanup:

kill -2 [pid]



View Logs
	1.	App logs
	1.	Console output:

RAG_Agent_vllm/src/app.log


	2.	Logger output:

RAG_Agent_vllm/src/app_rag.log


	2.	vllm logs

/root/autodl-tmp/models_vllm/vllm.log