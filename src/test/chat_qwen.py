import os
import re
from openai import OpenAI


def test_chat_qwen(system, prompt):
    qwen = OpenAI(base_url = "http://127.0.0.1:8000/v1", api_key= "EMPTY")
    completion = qwen.chat.completions.create(
    model="Qwen/Qwen2___5-72B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ],
    temperature=0,
    top_p=0.8,
    stream=False
    )
    response = completion.choices[0].message.content.strip()
    return response

if __name__ == "__main__":
    system = "You are a helpful assistant."
    prompt = "What is the capital of France? What is its coordinate "
    response = test_chat_qwen(system, prompt)
    print(response)
