from .base import VoiceAssistant
import transformers
import torch
import io
import base64
from openai import OpenAI
import soundfile as sf
import tempfile


class GPTAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.client = OpenAI()

    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ):
        # Write the audio data to an in-memory buffer in WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)  # Reset buffer position to the beginning

        # Read buffer as bytes and encode in base64
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

        completion = self.client.chat.completions.create(
            model=self.model_name,
            modalities=["text"],
            max_tokens=max_new_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who tries to help answer the user's question."},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}}]},
            ]
        )

        return completion.choices[0].message.content
    
    def generate_a2a(self, audio, max_new_tokens=2048):
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')
        completion = self.client.chat.completions.create(
            model=self.model_name,
            modalities=["text", "audio"],
            max_tokens=max_new_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who tries to help answer the user's question."},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}}]},
            ]
        )
        return completion.choices[0].message.content
    
    def asr(self, audio):
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)

        transcript = self.client.audio.transcriptions.create(
            model=self.asr_model_name,
            file=buffer,
        )

        return transcript['text']

    def tts(self, text):
        with self.client.audio.with_streaming_response.create(
            model=self.tts_model_name,
            text=text,
            response_format="wav"
        ) as response:
            # with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            #     response.stream_to_file(temp_file.name)
            #     audio_array, sample_rate = sf.read(temp_file.name)
            audio_array, sample_rate = sf.read(io.BytesIO(response.content))
                
        return audio_array, sample_rate
    

class GPT4oAssistant(GPTAssistant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "gpt-4o-audio-preview"
        self.asr_model_name = "gpt-4o-transcribe"
        self.tts_model_name = "gpt-4o-mini-tts"


class GPT4oMiniAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "gpt-4o-mini-audio-preview"
        self.asr_model_name = "gpt-4o-mini-transcribe"
        self.tts_model_name = "gpt-4o-mini-tts"