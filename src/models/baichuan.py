import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import ujson
from .base import VoiceAssistant
import tempfile
import soundfile as sf
import os
from huggingface_hub import snapshot_download
import sys
from .src_baichuan.generation import decode_wave_vocoder, GenerationAudioTokens

COSY_VOCODER = "src/models/src_baichuan/third_party/cosy24k_vocoder"
sys.path.append(os.path.join(COSY_VOCODER))


class BaichuanAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.sampling_rate = 24000
        self.role_prefix = {
            'system': '<B_SYS>',
            'user': '<C_Q>',
            'assistant': '<C_A>',
            'audiogen': '<audiotext_start_baichuan>'
        }
        self.load_model(**kwargs)
        self.model.training = False
        self.model.bind_processor(self.tokenizer, training=False, relative_path="/")
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)
        self.audiogen_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_start_token_id)
        self.audiogen_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_end_token_id)
        self.special_token_partten = re.compile(
            '<\|endoftext\|>|'
            '<audiogen_start_baichuan>|'
            '<audiogen_end_baichuan>'
        )
        # load the waveform vocoder
        from cosy24k_vocoder import Cosy24kVocoder
        self.vocoder = Cosy24kVocoder.from_pretrained(
            os.path.join(COSY_VOCODER, "hift.pt")
        ).cuda()

    def load_model(self, **kwargs):
        raise NotImplementedError

    def preprocess_messages(self, messages):
        text = ""
        for i, msg in enumerate(messages):
            text += self.role_prefix[msg['role']]
            text += msg['content']
        text += self.role_prefix["assistant"]
        return text

    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        g_history = []

        g_history.append({
            "role": "system",
            "content": "You are a helpful assistant who tries to help answer the user's question."
        })

        g_history.append({
            "role": "user",
            "content": self.audio_start_token + ujson.dumps({'path': temp_filename}, ensure_ascii=False) + self.audio_end_token
        })
        message = self.preprocess_messages(g_history)
        pret = self.model.processor([message])
        plen = pret.input_ids.shape[1]
        ret = self.model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(),
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            stop_strings=['<|endoftext|>'],
            do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
        )
        text_segment = self.tokenizer.decode(ret.sequences[0, plen:])
        full_text = re.sub(self.special_token_partten, '', text_segment)

        return full_text

    def generate_a2a(self, audio, max_new_tokens=500):
        # write input audio to a temp wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio['array'], audio['sampling_rate'], format='wav')
            path = tmp.name

        # build prompt: audio_in → audio_out
        prompt = (
            self.audio_start_token
            + ujson.dumps({'path': path}, ensure_ascii=False)
            + self.audio_end_token
            + self.audiogen_start_token
        )
        pret = self.model.processor([prompt])
        seq = pret.input_ids.cuda()

        # generate raw audio tokens
        audioret = GenerationAudioTokens.generate(
            self.model,
            seq,
            attention_mask=torch.ones_like(seq),
            past_key_values=None,
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.5,
            top_k=5, top_p=0.85,
            repetition_penalty=1.3,
            return_dict_in_generate=True,
        )
        # decode tokens to waveform
        wave_seg = decode_wave_vocoder(
            audioret.audios_sequences.clone(),
            self.vocoder,
            self.model
        )
        # clamp & convert to int16 numpy
        import numpy as np
        wg = (
            torch.clamp(wave_seg[0].squeeze(), -0.99, 0.99)
            .cpu()
            .numpy()
            * 32768.0
        ).astype(np.int16)
        return wg, self.sampling_rate


class BaichuanOmniAssistant(BaichuanAssistant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'baichuan_omni'
    def load_model(self, **kwargs):
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        if not os.path.exists(os.path.join(cache_dir, "Baichuan-Omni-1d5")):
            snapshot_download(
                repo_id="baichuan-inc/Baichuan-Omni-1d5",
                local_dir=os.path.join(cache_dir, "Baichuan-Omni-1d5"),
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(cache_dir, "Baichuan-Omni-1d5"), trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(cache_dir, "Baichuan-Omni-1d5"), trust_remote_code=True)

    def tts(self, text):
        prompt = text + self.audiogen_start_token
        pret = self.model.processor([prompt])
        seq = pret.input_ids.cuda()

        # Generate audio tokens
        audioret = GenerationAudioTokens.generate(
            self.model,
            seq,
            labels=None,
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            attention_mask=torch.ones_like(seq),
            past_key_values=None,
            max_new_tokens=700,
            num_beams=1,
            do_sample=True, 
            temperature=0.5,
            top_k=5, 
            top_p=0.85,
            num_return_sequences=1,
            repetition_penalty=1.3,
            return_dict_in_generate=True,
            tokenizer=self.tokenizer,
        )

        # Decode tokens to waveform
        wave_seg = decode_wave_vocoder(
            audioret.audios_sequences.clone(),
            self.vocoder,
            self.model
        )

        # Convert to int16 numpy array
        import numpy as np
        wg = (
            torch.clamp(wave_seg[0].squeeze(), -0.99, 0.99)
            .cpu()
            .numpy()
            * 32768.0
        ).astype(np.int16)

        return wg, self.sampling_rate

    def generate_at2t(
        self,
        audio,
        text,
        max_new_tokens=2048,
    ):
        # Create a temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        # Setup conversation history
        g_history = []

        # Add system message if needed
        g_history.append({
            "role": "system",
            "content": "You are a helpful assistant who tries to help answer the user's question."
        })

        # Add user message with both audio and text
        g_history.append({
            "role": "user",
            "content": self.audio_start_token + ujson.dumps({'path': temp_filename}, ensure_ascii=False) + self.audio_end_token + " " + text
        })
        
        # Preprocess messages to create input format
        message = self.preprocess_messages(g_history)
        
        # Process input through the model
        pret = self.model.processor([message])
        plen = pret.input_ids.shape[1]
        
        # Generate response
        ret = self.model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(),
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            stop_strings=['<|endoftext|>'],
            do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
        )
        
        # Decode generated tokens and clean up special tokens
        text_segment = self.tokenizer.decode(ret.sequences[0, plen:])
        full_text = re.sub(self.special_token_partten, '', text_segment)

        # Clean up temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass

        return full_text
    

class BaichuanAudioAssistant(BaichuanAssistant):
    def load_model(self, **kwargs):
        cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
        if not os.path.exists(os.path.join(cache_dir, "Baichuan-Audio-Instruct")):
            snapshot_download(
                repo_id="baichuan-inc/Baichuan-Audio-Instruct",
                local_dir=os.path.join(cache_dir, "Baichuan-Audio-Instruct"),
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(cache_dir, "Baichuan-Audio-Instruct"), trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(cache_dir, "Baichuan-Audio-Instruct"), trust_remote_code=True)