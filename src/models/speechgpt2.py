from .base import VoiceAssistant
import os
import io
import soundfile as sf
from huggingface_hub import snapshot_download
from transformers import HfArgumentParser

from .src_speechgpt2.inference import Inference
from .src_speechgpt2.mimo_qwen2_grouped import MIMOModelArguments


class SpeechGPT2(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'speechgpt2'
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        self.model_path = os.path.join(cache_dir, "SpeechGPT-2.0-preview-7B")
        if not os.path.exists(os.path.join(cache_dir, "SpeechGPT-2.0-preview-Codec")):
            snapshot_download(
                repo_id="fnlp/SpeechGPT-2.0-preview-Codec",
                local_dir=os.path.join(cache_dir, "SpeechGPT-2.0-preview-Codec"),
            )
        if not os.path.exists(self.model_path):
            snapshot_download(
                repo_id='fnlp/SpeechGPT-2.0-preview-7B',
                local_dir=self.model_path,
            )
        parser = HfArgumentParser((MIMOModelArguments,))
        self.model_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
        self.model_args.model_name_or_path = self.model_path
        self.model = Inference(
            model_path=self.model_path,
            model_args=self.model_args,
            codec_ckpt_path=os.path.join(cache_dir, "SpeechGPT-2.0-preview-Codec/sg2_codec_ckpt.pkl"),
            codec_config_path="./src/models/src_speechgpt2/Codec/config/sg2_codec_config.yaml"
        )

    def process_input(self, audio_input, text_input, task, mode=None, max_new_tokens=2048):
        if audio_input is not None:
            buffer = io.BytesIO()
            sf.write(buffer, audio_input['array'], audio_input['sampling_rate'], format='WAV')
            buffer.seek(0)
            input_data = buffer
        else:
            input_data = text_input

        return self.model.forward(
            task=task, input=input_data, text=text_input, mode=mode, max_new_tokens=max_new_tokens
        )

    def generate_a2t(self, audio, max_new_tokens=2048):
        self.model.process_greeting()
        response, _ = self.process_input(audio, None, task='thought', mode='s2t', max_new_tokens=max_new_tokens)
        return response
    
    def generate_t2t(self, text, max_new_tokens=2048):
        self.model.process_greeting()
        response, _ = self.process_input(None, text, task='thought', mode='t2t', max_new_tokens=max_new_tokens)
        return response
    
    def generate_t2a(self, text, max_new_tokens=2048):
        self.model.process_greeting()
        _, (sample_rate, wav) = self.process_input(None, text, task='thought', mode='t2s', max_new_tokens=max_new_tokens)
        return wav, sample_rate
    
    def generate_a2a(self, audio, max_new_tokens=2048):
        self.model.process_greeting()
        _, (sample_rate, wav) = self.process_input(audio, None, task='thought', mode='s2s', max_new_tokens=max_new_tokens)
        return wav, sample_rate
    
    def generate_at2t(self, audio, text, max_new_tokens=2048):
        self.model.process_greeting()
        response, _ = self.process_input(audio, text, task='thought', mode='st2t', max_new_tokens=max_new_tokens)
        return response
    
    def generate_at2a(self, audio, text, max_new_tokens=2048):
        self.model.process_greeting()
        _, (sample_rate, wav) = self.process_input(audio, text, task='thought', mode='st2s', max_new_tokens=max_new_tokens)
        return wav, sample_rate
    
    def tts(self, text):
        self.model.process_greeting()
        _, (sample_rate, wav) = self.process_input(None, text, task='tts')
        return wav, sample_rate
    
    def get_ppl(self, input, input_type: str):
        self.model.process_greeting()
        if input_type == 'text':
            mode = 't2t'
        elif input_type == 'audio':
            mode = 's2t'
            buffer = io.BytesIO()
            sf.write(buffer, input['array'], input['sampling_rate'], format='WAV')
            buffer.seek(0)
            input = buffer
        else:
            raise ValueError("Invalid input_type", input_type)
        return self.model.get_ppl([input], mode)