import time
from loguru import logger
from openai import OpenAI
import os

def generate_text_chat(client, *args, **kwargs):
    e = ''
    for _ in range(25):
        try:
            response = client.chat.completions.create(*args, **kwargs)
            time.sleep(0.5)
            if response is None:
                time.sleep(30)
                continue
            return response
        except Exception as e:
            logger.info(e)
            time.sleep(30)
    return None

class AIClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_URL'),
        )
        
    def generate(self, model, prompt):
        response = generate_text_chat(client=self.client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0)
        return response.choices[0].message.content.strip()