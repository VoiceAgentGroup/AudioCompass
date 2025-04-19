from .base import VoiceAssistant
import io
import base64
from openai import OpenAI
import soundfile as sf


class LocalAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

        models = self.client.models.list()
        self.model_name = models.data[0].id
        
    def get_response(self, content, max_tokens=2048):
        return self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who tries to help answer the user's question."
                },
                {
                    "role": "user", 
                    "content": content
                },
            ],
            extra_body={
                "prompt_logprobs": 0,
            },
        )
        
    def encode_audio(self, audio):
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')
        return encoded_string
        
    def generate_t2t(
        self,
        text,
        max_tokens=2048,
    ):
        content = [{"type": "text", "text": text}]
        completion = self.get_response(content, max_tokens)
        return completion.choices[0].message.content

    def generate_a2t(
        self,
        audio,
        max_tokens=2048,
    ):
        encoded_string = self.encode_audio(audio)
        content = [{"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}}]
        completion = self.get_response(content, max_tokens)
        return completion.choices[0].message.content

    def generate_at2t(
        self,
        audio,
        text,
        max_tokens=2048,
    ):
        encoded_string = self.encode_audio(audio)
        content = [
            {"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}},
            {"type": "text", "text": text}
        ]
        completion = self.get_response(content, max_tokens)
        return completion.choices[0].message.content
    
    def get_ppl(self, input, input_type: str):
        if input_type == 'text':
            content = [{"type": "text", "text": input}]
        elif input_type == 'audio':
            encoded_string = self.encode_audio(input)
            content = [{"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}}]
        else:
            raise ValueError("Invalid input_type", input_type)
        completion = self.get_response(content)
        ppl = -sum([list(token.values())[0]['logprob'] for token in completion.prompt_logprobs[1:]]) / len(completion.prompt_logprobs[1:])
        return ppl