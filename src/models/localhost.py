from .base import VoiceAssistant
import io
import base64
from openai import OpenAI
import soundfile as sf


class LocalAssistant(VoiceAssistant):
    def __init__(self):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

        models = self.client.models.list()
        self.model_name = models.data[0].id

    def generate_audio(
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
            max_tokens=max_new_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who tries to help answer the user's question."},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}}]},
            ],
            extra_body={
                "prompt_logprobs": 1,
            },
        )
        
        return completion.choices[0].message.content, completion.prompt_logprobs
    

    def generate_mixed(
        self,
        audio,
        text,
        max_new_tokens=2048,
    ):
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)

        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')
        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who tries to help answer the user's question."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}},
                        {"type": "text", "text": text}
                    ]
                },
            ],
        )
        return completion.choices[0].message.content