from .base import VoiceAssistant
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class Qwen3Assistant_T(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'qwen3-8b-thinking'
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(cache_dir, "Qwen3-8B"))
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(cache_dir, "Qwen3-8B"),
            torch_dtype="auto",
            device_map="sequential"
        )

    def generate_t2t(self, text):
        messages = [
            {"role": "user", "content": text},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return response

class Qwen3Assistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'qwen3-8b'
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(cache_dir, "Qwen3-8B"))
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(cache_dir, "Qwen3-8B"),
            torch_dtype="auto",
            device_map="sequential"
        )

    def generate_t2t(self, text):
        messages = [
            {"role": "user", "content": text},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return response
