import os
import torchaudio
import torch
from tqdm import tqdm
from loguru import logger
import json
from ..base import BaseBenchmark
from src.transcriptors import Paraformer, WhisperLargeV3
from .evaluate import verification, process_one_wer

class SeedTTSEval(BaseBenchmark):
    def __init__(self, split, data_dir="datas/seedtts_testset", cache_dir='cache', **kwargs):
        self.name = 'seed-tts-eval'
        self.check_split(split)
        self.split = split
        self.data_dir = os.path.join(cache_dir, data_dir)
        logger.add(f'log/{self.name}-{self.split}.log', rotation='50 MB')
        self.dataset = self.load_data()
        self.offline = kwargs.get('offline', False)
        self.transcriptor = WhisperLargeV3(**kwargs) if split == 'en' else Paraformer(**kwargs)
        self.wavlm_path = os.path.join(cache_dir, 'models', 'wavlm_large_finetune.pth')
        
    def check_split(self, split):
        available_split = ['en', 'zh', 'hard']
        if split not in available_split:
            raise ValueError("Split should be one of " + available_split)
        
    def load_data(self):
        logger.info("Preparing data ...")
        if self.split == 'en' or self.split == 'zh':
            meta_path = os.path.join(self.data_dir, self.split, 'meta.lst')
            wav_dir = os.path.join(self.data_dir, self.split, 'wavs')
        else:
            meta_path = os.path.join(self.data_dir, 'zh/hardcase.lst')
            wav_dir = os.path.join(self.data_dir, 'zh/wavs')
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        
        dataset = []
        for line in tqdm(lines):
            if len(line.strip().split('|')) == 5:
                utt, prompt_text, prompt_wav, infer_text, infer_wav = line.strip().split('|')
            elif len(line.strip().split('|')) == 4:
                utt, prompt_text, prompt_wav, infer_text = line.strip().split('|')
            elif len(line.strip().split('|')) == 2:
                utt, infer_text = line.strip().split('|')
            elif len(line.strip().split('|')) == 3:
                utt, infer_text, prompt_wav = line.strip().split('|')
                if utt.endswith(".wav"):
                    utt = utt[:-4]
            if not os.path.exists(os.path.join(wav_dir, utt + '.wav')):
                continue
            dataset.append({
                'infer_text': infer_text,
                'ref_wav_path': os.path.join(wav_dir, utt + '.wav'),
            })
            
        return dataset
    
    def generate(self, model):
        logger.info("Generating results ...")
        results = []
        
        for idx, data in tqdm(enumerate(self.dataset)):
            try:
                response_audio, sample_rate = model.tts(data['infer_text'])
                transcription = self.transcriptor.inference(response_audio)
                results.append({
                    'idx': idx,
                    'infer_text': data['infer_text'],
                    'tts_wav': response_audio,
                    'sample_rate': sample_rate,
                    'transcription': transcription,
                    'ref_wav_path': data['ref_wav_path'],
                })
                logger.info(f"Transcription: {transcription}")
                logger.info('====================================')
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
            
        return results
    
    def evaluate(self, results):
        logger.info("Evaluating results ...")
        
        total_wer = 0
        total_sim = 0
        for result in tqdm(results):
            raw_truth, raw_hypo, wer, subs, dele, inse = process_one_wer(self.split, result['transcription'], result['infer_text'])
            total_wer += wer
            # sim, model = verification(model_name='wavlm_large', wav1=result['tts_wav_path'], wav2=result['ref_wav_path'], checkpoint=self.wavlm_path, offline=self.offline)
            # total_sim += sim.cpu().item()
        avg_wer = total_wer / len(results)
        # avg_sim = total_sim / len(results)
        
        # return {'wer': avg_wer, 'sim': avg_sim}
        return {'wer': avg_wer}
    
    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, self.name)
        wav_dir = os.path.join(output_dir, 'tts-wavs')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        for result in results:
            wav_path = os.path.join(wav_dir, f"{model_name}-{self.split}-{result['idx']}.wav")
            wav = torch.tensor(result['tts_wav'], dtype=torch.float32) if result['tts_wav'].ndim == 2 else torch.tensor(result['tts_wav'], dtype=torch.float32).unsqueeze(0)
            torchaudio.save(wav_path, wav, result['sample_rate'])
            result['tts_wav_path'] = wav_path
            result.pop('tts_wav')
            result.pop('sample_rate')
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-{self.split}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Generated results saved to {output_file}.")
        
    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        evaluated_result = self.evaluate(generated_results)
        return evaluated_result