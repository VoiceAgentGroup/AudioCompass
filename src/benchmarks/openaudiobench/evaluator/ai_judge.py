import os
import httpx
from openai import OpenAI
from src.api import generate_text_chat

class AIJudge:

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url="https://172.203.11.191:3826/v1",
            http_client=httpx.Client()
        )
    def generate(self, prompt):
        response = generate_text_chat(client=self.client,
            model='gpt-4o-mini',
            messages=[{"role": "system",
                        "content": "You are a helpful assistant who tries to help answer the user's question."},
                        {"role": "user", "content": prompt}],
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0)
        return response.choices[0].message.content.strip()