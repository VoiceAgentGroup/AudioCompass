from .base import VoiceAssistant
import os
import io
import soundfile as sf
from huggingface_hub import snapshot_download
from transformers import HfArgumentParser

from .src_speechgpt2.inference import Inference
from .src_speechgpt2.mimo_qwen2_grouped import MIMOModelArguments


class SpeechGPT2(VoiceAssistant):
    def __init__(self):
        self.model_name = 'speechgpt2'
        self.model_path = "./cache/SpeechGPT-2.0-preview-7B"
        if not os.path.exists("./cache/SpeechGPT-2.0-preview-Codec"):
            snapshot_download(
                repo_id="fnlp/SpeechGPT-2.0-preview-Codec",
                local_dir="./cache/SpeechGPT-2.0-preview-Codec",
            )
        if not os.path.exists("./cache/SpeechGPT-2.0-preview-7B"):
            snapshot_download(
                repo_id='fnlp/SpeechGPT-2.0-preview-7B',
                local_dir="./cache/SpeechGPT-2.0-preview-7B",
            )
        parser = HfArgumentParser((MIMOModelArguments,))
        self.model_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
        self.model_args.model_name_or_path = self.model_path
        self.model = Inference(
            model_path=self.model_path,
            model_args=self.model_args,
            codec_ckpt_path="./cache/SpeechGPT-2.0-preview-Codec/sg2_codec_ckpt.pkl",
            codec_config_path="./src/models/src_speechgpt2/Codec/config/sg2_codec_config.yaml"
        )

    def process_input(self, audio_input, text_input, mode):
        try:
            # Handle audio input
            if audio_input is not None:
                buffer = io.BytesIO()
                sf.write(buffer, audio_input['array'], audio_input['sample_rate'], format='WAV')
                buffer.seek(0)
                input_data = buffer
            else:
                input_data = text_input

            return self.model.forward(
                task='thought', input=input_data, text=text_input, mode=mode
            )

        except Exception as e:
            return f"Error: {str(e)}", None, None

    def generate_a2t(self, audio):
        response, _ = self.process_input(audio, None, 's2t')
        self.model.clear_history()
        return response, None
    
    def generate_t2t(self, text):
        response, _ = self.process_input(None, text, 't2t')
        self.model.clear_history()
        return response, None
    
    def generate_t2s(self, text):
        response, wav = self.process_input(None, text, 't2s')
        self.model.clear_history()
        return response, wav  # wav: tuple (sample_rate, array)
    
    def generate_s2s(self, audio):
        response, wav = self.process_input(audio, None, 's2s')
        self.model.clear_history()
        return response, wav  # wav: tuple (sample_rate, array)