from .base import VoiceAssistant

import torch, math
from transformers import MoshiForConditionalGeneration, AutoFeatureExtractor, AutoTokenizer
from huggingface_hub import snapshot_download
import librosa
import os
import torch
import math
import librosa
import torch.nn.functional as F

class MoshiAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'moshi'
        self.device = 'cuda'
        self.dtype = torch.float16
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        model_path = os.path.join(cache_dir, "hf-moshiko")
        if not os.path.exists(model_path):
            snapshot_download(
                repo_id="kmhf/hf-moshiko",
                local_dir=model_path,
            )
        self.model = MoshiForConditionalGeneration.from_pretrained(model_path, device_map=self.device, torch_dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ):
        audio = audio['array']
        audio = librosa.resample(audio, orig_sr=16000, target_sr=self.feature_extractor.sampling_rate)

        user_input_values = self.feature_extractor(raw_audio=audio, sampling_rate=self.feature_extractor.sampling_rate,
                                              return_tensors="pt").to(device=self.device, dtype=self.dtype)

        # prepare moshi input values - we suppose moshi didn't say anything while the user spoke
        moshi_input_values = torch.zeros_like(user_input_values.input_values)

        ratio = self.model.config.audio_encoder_config.frame_rate / self.model.config.sampling_rate

        # prepare moshi input ids - we suppose moshi didn't say anything while the user spoke
        num_tokens = math.ceil(moshi_input_values.shape[-1] * ratio)
        input_ids = torch.ones((1, num_tokens), device=self.device, dtype=torch.int64) * self.tokenizer.encode("<pad>")[0]

        output = self.model.generate(input_ids=input_ids, user_input_values=user_input_values.input_values,
                                moshi_input_values=moshi_input_values, max_new_tokens=max_new_tokens, return_audio_waveforms=False)

        text_tokens = output.cpu().numpy()

        response = self.tokenizer.batch_decode(text_tokens, skip_special_tokens=True)[0]

        return response
    
    def generate_a2a(
        self,
        audio,
        max_new_tokens=1024,
    ):
        audio = audio['array']
        audio = librosa.resample(audio, orig_sr=16000, target_sr=self.feature_extractor.sampling_rate)

        user_input_values = self.feature_extractor(raw_audio=audio, sampling_rate=self.feature_extractor.sampling_rate,
                                              return_tensors="pt").to(device=self.device, dtype=self.dtype)

        # prepare moshi input values - we suppose moshi didn't say anything while the user spoke
        moshi_input_values = torch.zeros_like(user_input_values.input_values)

        ratio = self.model.config.audio_encoder_config.frame_rate / self.model.config.sampling_rate

        # prepare moshi input ids - we suppose moshi didn't say anything while the user spoke
        num_tokens = math.ceil(moshi_input_values.shape[-1] * ratio)
        input_ids = torch.ones((1, num_tokens), device=self.device, dtype=torch.int64) * self.tokenizer.encode("<pad>")[0]

        output = self.model.generate(input_ids=input_ids, user_input_values=user_input_values.input_values,
                                moshi_input_values=moshi_input_values, max_new_tokens=1024)

        audio_waveforms = output.audio_sequences

        return audio_waveforms.float().squeeze().detach().cpu().numpy(), 24000

    def generate_t2t(self, text, max_new_tokens=2048):
        inputs = self.tokenizer(text, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids, max_length=max_new_tokens)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
    
   

    def get_ppl(self, audio, input_type: str):
        if input_type != "audio":
            raise ValueError(f"Unsupported input_type: {input_type}")

        audio = audio['array']
        audio = librosa.resample(audio, orig_sr=16000, target_sr=self.feature_extractor.sampling_rate)

        user_input_values = self.feature_extractor(raw_audio=audio, sampling_rate=self.feature_extractor.sampling_rate,
                                              return_tensors="pt").to(device=self.device, dtype=self.dtype)

        # prepare moshi input values - we suppose moshi didn't say anything while the user spoke
        moshi_input_values = torch.zeros_like(user_input_values.input_values)

        ratio = self.model.config.audio_encoder_config.frame_rate / self.model.config.sampling_rate

        # prepare moshi input ids - we suppose moshi didn't say anything while the user spoke
        num_tokens = math.ceil(moshi_input_values.shape[-1] * ratio)
        input_ids = torch.ones((1, num_tokens), device=self.device, dtype=torch.int64) * self.tokenizer.encode("<pad>")[0]

        output = self.model.generate(input_ids=input_ids, user_input_values=user_input_values.input_values,
                                moshi_input_values=moshi_input_values, max_new_tokens=1024)
        
        print(output.keys())
        
        return output.loss
        
