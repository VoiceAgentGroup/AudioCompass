from .base import VoiceAssistant
from transformers import AutoModel, AutoTokenizer
from transformers import WhisperFeatureExtractor, AutoTokenizer
from loguru import logger
import torch
import os
import numpy as np
import uuid
import torchaudio
from .src_glm.speech_tokenizer.utils import extract_speech_token
from .src_glm.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from .src_glm.flow_inference import AudioDecoder
from .src_glm.audio_process import AudioStreamProcessor


class GLMAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'glm'
        model_path = 'THUDM/glm-4-voice-9b'
        cache_dir = os.path.join(kwargs.get('cache_dir', './cache'), 'models')
        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=None,
            device_map={"": 0},
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("THUDM/glm-4-voice-tokenizer", cache_dir=cache_dir)
        self.whisper_model = WhisperVQEncoder.from_pretrained("THUDM/glm-4-voice-tokenizer", cache_dir=cache_dir).eval().to("cuda")
        
        # Initialize AudioDecoder for generate_a2a method
        flow_config = os.path.join(cache_dir, 'glm-4-voice-decoder', 'config.yaml')
        flow_checkpoint = os.path.join(cache_dir, 'glm-4-voice-decoder', 'flow.pt')
        hift_checkpoint = os.path.join(cache_dir, 'glm-4-voice-decoder', 'hift.pt')
        self.audio_decoder = None
        self.audio_processor = None
        if os.path.exists(flow_config) and os.path.exists(flow_checkpoint) and os.path.exists(hift_checkpoint):
            self.audio_decoder = AudioDecoder(
                config_path=flow_config,
                flow_ckpt_path=flow_checkpoint,
                hift_ckpt_path=hift_checkpoint,
                device="cuda"
            )
            self.audio_processor = AudioStreamProcessor()

    def generate_a2t(
        self,
        audio,
        max_new_tokens=4096,
    ):
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [tuple([torch.from_numpy(audio['array']).unsqueeze(0), audio['sampling_rate']])]
        )[0]
        assert len(audio_tokens) != 0
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        user_input = audio_tokens
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = inputs.to('cuda')

        rtn = self.glm_model.generate(**inputs, max_new_tokens=max_new_tokens)[:, inputs.input_ids.size(1):]
        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for item in rtn[0]:
            if item < audio_offset:
                text_tokens.append(item)
        # logger.info(text_tokens)
        return self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)

    def generate_t2t(
        self,
        text,
    ):
        history = []
        history.append({"role": "user", "content": text})
        user_input = text
        system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        # system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in text tokens only."
        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = inputs.to('cuda')
        rtn = self.glm_model.generate(**inputs, max_new_tokens=4096)[:, inputs.input_ids.size(1):]
        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for item in rtn[0]:
            if item < audio_offset:
                text_tokens.append(item)
        # logger.info(text_tokens)
        return self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)

    def generate_a2a(
        self,
        audio,
        max_new_tokens=4096,
    ):
        if self.audio_decoder is None:
            raise ValueError("Audio decoder not initialized. Please provide 'flow_path' in initialization.")
            
        # Extract audio tokens from the input audio
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [tuple([torch.from_numpy(audio['array']).unsqueeze(0), audio['sampling_rate']])]
        )[0]
        assert len(audio_tokens) != 0
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        user_input = audio_tokens
        
        # Set system prompt for audio to audio generation
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
        
        # Prepare model inputs
        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = inputs.to('cuda')
        
        # Generate response tokens
        rtn = self.glm_model.generate(**inputs, max_new_tokens=max_new_tokens)[:, inputs.input_ids.size(1):]
        
        # Separate text and audio tokens
        text_tokens = []
        audio_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for item in rtn[0]:
            if item >= audio_offset:
                audio_tokens.append(item - audio_offset)
            else:
                text_tokens.append(item)
        
        # Convert text tokens to text output
        text_output = self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
        
        # Convert audio tokens to audio output if we have audio tokens
        audio_output = None
        if len(audio_tokens) > 0:
            audio_token_tensor = torch.tensor(audio_tokens, device="cuda").unsqueeze(0)
            this_uuid = str(uuid.uuid4())
            
            # Generate audio waveform using the decoder
            audio_waveform, _ = self.audio_decoder.token2wav(
                audio_token_tensor, 
                uuid=this_uuid,
                finalize=True
            )
            
            # Convert to numpy array
            audio_output = audio_waveform.cpu().numpy()[0]
        
        return audio_output, 22050