from .base import VoiceAssistant
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import os


class Qwen2Assistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'qwen2'
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=cache_dir)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct",
                                                                        device_map="cuda",
                                                                        cache_dir=cache_dir, torch_dtype='auto')

    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ):
        assert audio['sampling_rate'] == 16000
        audio = audio['array']
        content = [{"type": "audio", "audio_url": 'xxx'}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = [audio]
        inputs = self.processor(text=inputs, audios=audios, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=max_new_tokens)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    def generate_t2t(
        self,
        text,
    ):
        content = [{"type": "text", "text": text}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=inputs, audios=None, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=2048)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = \
        self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
