from .base import VoiceAssistant
import io
import base64
import os
from openai import OpenAI
import soundfile as sf
import numpy as np

class QwenOmniAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = "qwen-omni-turbo"
    
    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ):
        # Write the audio data to an in-memory buffer in WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)

        # Read buffer as bytes and encode in base64
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:;base64,{encoded_string}",
                                "format": "wav",
                            },
                        }
                    ]
                }
            ],
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
        )
        
        collected_messages = []
        
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                collected_messages.append(content)
        
        full_response = "".join(collected_messages)
        
        return full_response


    def generate_at2t(
        self,
        audio,
        text,
        max_new_tokens=2048,
    ):
        # Write the audio data to an in-memory buffer in WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)

        # Read buffer as bytes and encode in base64
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:;base64,{encoded_string}",
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": text,
                        }
                    ]
                }
            ],
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
        )
        
        collected_messages = []
        
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                collected_messages.append(content)
        
        full_response = "".join(collected_messages)
        
        return full_response


    def generate_a2a(
        self,
        audio,
        max_new_tokens=2048,
    ):
        # Write the audio data to an in-memory buffer in WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)

        # Read buffer as bytes and encode in base64
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:;base64,{encoded_string}",
                                "format": "wav",
                            },
                        }
                    ]
                }
            ],
            modalities=["text", "audio"],
            audio={"voice": "Ethan", "format": "wav"},
            stream=True,
            stream_options={"include_usage": True},
        )
        
        audio_string = ""
        for chunk in completion:
            if chunk.choices:
                if hasattr(chunk.choices[0].delta, "audio"):
                    try:
                        audio_string += chunk.choices[0].delta.audio["data"]
                    except Exception as e:
                        pass
            else:
                print(chunk.usage)

        wav_bytes = base64.b64decode(audio_string)
        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)

        return audio_np, 24000