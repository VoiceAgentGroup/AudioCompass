from .base import VoiceAssistant
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.models.src_step_audio.tokenizer import StepAudioTokenizer
from src.models.src_step_audio.tts import StepAudioTTS
from src.models.src_step_audio.utils import load_audio
from huggingface_hub import snapshot_download
import os
import tempfile
import soundfile as sf


class StepAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'step_audio'
        
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        tokenizer_path = os.path.join(cache_dir, "Step-Audio-Tokenizer")
        model_path = os.path.join(cache_dir, "Step-Audio-Chat")
        tts_path = os.path.join(cache_dir, 'Step-Audio-TTS-3B')
        
        if not os.path.exists(tokenizer_path):
            snapshot_download(
                repo_id="stepfun-ai/Step-Audio-Tokenizer",
                local_dir=tokenizer_path,
            )
        if not os.path.exists(model_path):
            snapshot_download(
                repo_id="stepfun-ai/Step-Audio-Chat",
                local_dir=model_path,
            )
        if not os.path.exists(tts_path):
            snapshot_download(
                repo_id="stepfun-ai/Step-Audio-Chat",
                local_dir=tts_path,
            )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.encoder = StepAudioTokenizer(tokenizer_path)
        # self.decoder = StepAudioTTS(tts_path, self.encoder)

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        messages = [{"role": "user", "content": {"type": "audio", "audio": temp_filename}}]
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(text_with_audio, return_tensors="pt")
        outputs = self.llm.generate(
            token_ids, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True
        )
        output_token_ids = outputs[:, token_ids.shape[-1]: -1].tolist()[0]
        output_text = self.llm_tokenizer.decode(output_token_ids)
        return output_text
    
    def generate_a2a(
        self,
        audio,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        messages = [{"role": "user", "content": {"type": "audio", "audio": temp_filename}}]
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(text_with_audio, return_tensors="pt")
        outputs = self.llm.generate(
            token_ids, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True
        )
        output_token_ids = outputs[:, token_ids.shape[-1]: -1].tolist()[0]
        output_text = self.llm_tokenizer.decode(output_token_ids)
        output_audio, sample_rate = self.decoder(output_text, "Tingting")
        return output_audio, sample_rate
    
    def generate_at2t(
        self,
        audio,
        text,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        messages = [{"role": "user", "content": [{"type": "audio", "audio": temp_filename}, {"type": "text", "text": text}]}]
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(text_with_audio, return_tensors="pt")
        outputs = self.llm.generate(
            token_ids, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True
        )
        output_token_ids = outputs[:, token_ids.shape[-1]: -1].tolist()[0]
        output_text = self.llm_tokenizer.decode(output_token_ids)
        return output_text

    def encode_audio(self, audio_path):
        audio_wav, sr = load_audio(audio_path)
        audio_tokens = self.encoder(audio_wav, sr)
        return audio_tokens

    def apply_chat_template(self, messages: list):
        text_with_audio = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio += f"<|BOT|>{role}\n{content}<|EOT|>"
            elif isinstance(content, dict):
                if content["type"] == "text":
                    text_with_audio += f"<|BOT|>{role}\n{content['text']}<|EOT|>"
                elif content["type"] == "audio":
                    audio_tokens = self.encode_audio(content["audio"])
                    text_with_audio += f"<|BOT|>{role}\n{audio_tokens}<|EOT|>"
            elif content is None:
                text_with_audio += f"<|BOT|>{role}\n"
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        if not text_with_audio.endswith("<|BOT|>assistant\n"):
            text_with_audio += "<|BOT|>assistant\n"
        return text_with_audio
    
    def get_ppl(self, input, input_type: str):
        if input_type == 'text':
            messages = [{"role": "user", "content": input}]
        elif input_type == 'audio':
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
                temp_filename = temp_file.name
                # Write the audio data to the file
                sf.write(temp_file.name, input['array'], input['sampling_rate'], format='wav')

            messages = [{"role": "user", "content": {"type": "audio", "audio": temp_filename}}]
        else:
            raise ValueError("Invalid input_type", input_type)
        
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(text_with_audio, return_tensors="pt")
        
        output = self.llm.forward(token_ids, labels=token_ids)
        loss = output.loss
        
        return loss.item()
    
    def tts(self, text):
        output_audio, sample_rate = self.decoder(text, "Tingting")
        return output_audio, sample_rate

    def tts_clone(self, text, prompt_audio_path, prompt_text):
        clone_speaker = {
            "wav_path": prompt_audio_path,
            "speaker": "custom_voice",
            "prompt_text": prompt_text,
        }
        output_audio, sample_rate = self.decoder(text, "", clone_speaker)
        return output_audio, sample_rate
        