import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class WhisperLargeV3:
    def __init__(self, **kwargs):
        if kwargs.get('offline', False):
            cache_dir = kwargs.get('cache_dir', 'cache')
            self.model_id = os.path.join(cache_dir, 'whisper-large-v3')
        else:
            self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True,
        )
        
    def inference(self, audio):
        if audio.ndim > 1:
            audio = audio[0, :]
        result = self.pipe(audio)["text"]
        return result
