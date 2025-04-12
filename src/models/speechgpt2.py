from .base import VoiceAssistant
import os
from huggingface_hub import snapshot_download
from transformers import HfArgumentParser

from .src_speechgpt2.inference import Inference
from .src_speechgpt2.mimo_qwen2_grouped import MIMOModelArguments


class SpeechGPT2(VoiceAssistant):
    def __init__(self):
        self.model_name = 'speechgpt2'
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
        self.model_args.model_name_or_path = self.args.model_path
        self.model = Inference(
            model_path="./cache/SpeechGPT-2.0-preview-7B",
            model_args=self.model_args,
            codec_ckpt_path="./cache/SpeechGPT-2.0-preview-Codec/sg2_codec_ckpt.pkl",
            codec_config_path="./src/models/src_speechgpt2/Codec/config/sg2_codec_config.yaml"
        )

    def generate_audio(self, audio, max_new_tokens=2048):
        pass