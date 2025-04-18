from .base import VoiceAssistant
import torch
import os
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


class PhiAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        model_path = 'microsoft/Phi-4-multimodal-instruct'
        kwargs['torch_dtype'] = torch.bfloat16

        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        self.model_name = 'phi'
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
            device_map='cuda',
            cache_dir=cache_dir
        )

        self.generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'
        print(self.generation_config)

    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ):
        chat = [{'role': 'user', 'content': f'<|audio_1|>'}]
        prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
        if prompt.endswith('<|endoftext|>'):
            prompt = prompt.rstrip('<|endoftext|>')
        audio = [audio['array'], audio['sampling_rate']]
        inputs = self.processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_logits_to_keep=0,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
