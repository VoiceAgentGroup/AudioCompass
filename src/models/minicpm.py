from .base import VoiceAssistant
from transformers import AutoModel, AutoTokenizer
import transformers
import torch
import soundfile as sf
import os


class MiniCPMAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'minicpm'
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        self.model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=False,
            cache_dir=cache_dir,
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

        self.sys_prompt = self.model.get_sys_prompt(mode='audio_assistant', language='en')

    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ): 
        user_question = {'role': 'user', 'content': [audio['array']]}
        msgs = [self.sys_prompt, user_question]
        
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=max_new_tokens,
            use_tts_template=True,
            generate_a2t=False,
            temperature=0.3,
        )

        return res
