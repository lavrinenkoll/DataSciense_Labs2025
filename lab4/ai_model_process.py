import os
import time

import transformers
import torch
from groq import Groq
from dotenv import load_dotenv


def get_response_api(prompt: str, system_prompt: str = None) -> str:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    client = Groq(api_key=api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.1-8b-instant"
    )

    return response.choices[0].message.content
