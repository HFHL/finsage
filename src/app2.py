import os
import sys
import json
import logging
import yaml
import traceback
import time
import uuid
from functools import wraps
from flask import Flask, request, Response, stream_with_context, render_template, session
from flask_cors import CORS
from openai import OpenAI
# Load audio file with Whisper
import whisper

from utils.ragManager import RAGManager
from utils.vllmChatService import ChatService

from gpu_log import log_gpu_usage

os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用于Flask session
CORS(app)

class GlobalResponseHandler:
    @staticmethod
    def success(data=None, message="Success", status_code=200, response_time=None):
        return GlobalResponseHandler._create_response("success", message, data, status_code, response_time)

    @staticmethod
    def error(message="An error occurred", data=None, status_code=400, response_time=None):
        return GlobalResponseHandler._create_response("error", message, data, status_code, response_time)

    @staticmethod
    def _create_response(status, message, data, status_code, response_time):
        response = {
            "status": status,
            "message": message,
            "data": data,
            "response_time": response_time
        }
        response_json = json.dumps(response)
        return Response(response=response_json, status=status_code, mimetype='application/json')

    @staticmethod
    def stream_response(generate_func):
        return Response(stream_with_context(generate_func()), content_type='text/event-stream')


def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds")
        return result

    return wrapper


@timing_decorator
def warm_up(config):
    logger.info("Starting warm up")
    try:
        llm = OpenAI(
            api_key="EMPTY",
            base_url=config['ollama_base_url'],
        )
        response = llm.chat.completions.create(
            model=config['llm'],
            messages=[
                {"role": "system", "content": "Warm up"},
                {"role": "user", "content": "Warm up"},
            ]
        )
        logger.info(f"Response from warm-up request: {response.choices[0].message.content}")
    except Exception as e:
        logger.error(f'Warm-up request failed: {str(e)}')

@app.route('/api_chat', methods=['POST'])
def api_chat():
    try:
        data = request.json
        question = data.get('question')
        internal_input = data.get('internal_input', None)  # Get internal assistant input
        interrupt_index = data.get('interrupt_index', None)  # Get interrupt index
        session_id = session.get('session_id1')

        if not question:
            return GlobalResponseHandler.error(message="Question not provided")        

        response =  Response(
            stream_with_context(
                chat_service.generate_response_stream(
                    question,
                    session_id,
                    internal_input,
                    interrupt_index
                )
            ),
            content_type='text/event-stream'
        )
        

        return response
    except Exception as e:
        logger.error(f"An error occurred in /chat endpoint: {str(e)}")
        logger.error("".join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
        return GlobalResponseHandler.error(message=str(e))

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"An unexpected error occurred: {str(e)}")
    return GlobalResponseHandler.error(message=f"Internal Server Error: {str(e)}")


@app.route('/test_api_chat')
def test_api_chat():
    session_id = str(uuid.uuid4())
    session['session_id1'] = session_id
    _ = chat_service.get_or_create_chat_manager(session_id)
    return render_template('test_api.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204


# Author: hhl
# Date: 2024/10/11
# Description: This function handles audio file uploads from the user, transcribes the audio using OpenAI's Whisper model, and returns the transcription result.
@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        # Save audio chunk from request
        audio_file = request.files['audio']
        temp_audio_path = f"/tmp/{uuid.uuid4()}.webm"
        audio_file.save(temp_audio_path)

        
        model = whisper.load_model("base")
        options = {"language": "Chinese"}
        result = model.transcribe(temp_audio_path, **options)

        # Remove temporary audio file
        os.remove(temp_audio_path)

        # Return the transcription result
        return GlobalResponseHandler.success(data={"transcription": result["text"]})
    except Exception as e:
        logger.error(f"An error occurred during audio upload: {str(e)}")
        logger.error("".join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
        return GlobalResponseHandler.error(message=str(e))


if __name__ == "__main__":

    log_gpu_usage('server started!!!💥')

    config_path = os.getenv('CONFIG_PATH', '../config/config_vllm.yaml')
    config = load_config(config_path)

    log_level = config.get('log_level', 'INFO')
    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        filename='app_rag.log', 
        filemode='w', 
        level=numeric_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    print(f'log level {log_level}, numeric level: {numeric_level}, log file: app_rag.log')


    collections = {'lotus': 10, 'lotus_car_stats': 0, 'lotus_brand_info': 0}

    # Creat Beans
    rag_manager = RAGManager(config=config, collections=collections)
    chat_service = ChatService(config=config, rag_manager=rag_manager)

    log_gpu_usage('warm up')
    warm_up(config)
    log_gpu_usage('warm up finished')

    # app.run(host='0.0.0.0', port=int(os.getenv('PORT', 6006)))
    app.run(host='0.0.0.0',port=int(os.getenv('PORT', 6005)))