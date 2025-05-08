from loguru import logger
from tqdm import tqdm
import json
import os
import pandas as pd
from .base import BaseBenchmark
import torchaudio
import jiwer


class CommonVoice(BaseBenchmark):
    def __init__(self, split, data_dir="datas/cv-corpus-20.0-2024-12-06", cache_dir='cache', **kwargs):
        self.name = 'commonvoice'
        self.check_split(split)
        self.split = split
        self.data_dir = os.path.join(cache_dir, data_dir, split)
        logger.add(f'log/{self.name}-{self.split}.log', rotation='50MB')
        self.dataset = self.load_data(**kwargs)
        
    def check_split(self, split):
        available_split = ['zh-CN',]
        if split not in available_split:
            raise ValueError("Split should be one of " + available_split)
        
    def load_data(self, **kwargs):
        logger.info("Preparing data ...")
        meta_path = os.path.join(self.data_dir, 'test.tsv')
        df = pd.read_csv(meta_path, sep='\t')
        
        dataset = []
        for i in tqdm(range(len(df))):
            audio_path = os.path.join(self.data_dir, 'clips', df.iloc[0, 1])
            wav, sample_rate = torchaudio.load(audio_path)
            audio = {
                'array': wav.squeeze(0).numpy(),
                'sampling_rate': sample_rate
            }
            sentence = df.iloc[0, 3]
            dataset.append({
                'audio': audio,
                'ref_text': sentence
            })
            
        return dataset
    
    def generate(self, model):
        logger.info("Generating results ...")
        results = []
        for idx, item in enumerate(tqdm(self.dataset, total=len(self.dataset))):
            try:
                response = model.asr(item['audio'])
                logger.info(response)
                results.append({
                    'idx': idx,
                    'res_text': response,
                    'ref_text': item['ref_text']
                })
                logger.info('====================================')
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
            
    def evaluate(self, results):
        logger.info("Evaluating results ...")
        total_er = 0
        if self.split == 'zh-CN':
            for result in tqdm(results):
                cer = jiwer.cer(result['ref_text'], results['res_text'])
                total_er += cer
            return {'cer': total_er / len(results)}
        else:
            for result in tqdm(results):
                wer = jiwer.wer(result['ref_text'], results['res_text'])
                total_er += wer
            return {'wer': total_er / len(results)}
        
    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-{self.split}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Generated results saved to {output_file}.")
        
    def run(self, model, output_dir):
        return super().run(model, output_dir)