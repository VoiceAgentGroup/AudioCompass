from funasr import AutoModel
import zhconv
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class Paraformer:
    def __init__(self, **kwargs):
        if kwargs.get('offline', False):
            cache_dir = os.path.join(kwargs.get('cache_dir', 'cache'), 'models')
            model_path = os.path.join(cache_dir, 'paraformer-zh')
            self.model = AutoModel(model=model_path, device=device, disable_update=True)
        else:
            self.model = AutoModel(model="paraformer-zh", device=device)
        
    def inference(self, audio):
        res = self.model.generate(input=audio, batch_size_s=300)
        transcription = res[0]["text"]
        transcription = zhconv.convert(transcription, 'zh-cn')
        return transcription