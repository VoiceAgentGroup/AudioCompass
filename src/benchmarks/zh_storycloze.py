from loguru import logger
from tqdm import tqdm
import json
import os
import numpy as np
from .base import BaseBenchmark
import torchaudio
import torch


class zhStoryCloze(BaseBenchmark):
    def __init__(self, data_dir="datas/zh-storycloze", cache_dir='cache', **kwargs):
        self.name = 'zh-storycloze'
        self.data_dir = os.path.join(cache_dir, data_dir)
        logger.add(f'log/{self.name}.log', rotation='50 MB')
        self.dataset = self.load_data()
        
    def concat_audio(self, prefix_path, suffix_paths):
        audio_group = []

        prefix_path = os.path.join(self.data_dir, prefix_path)
        if not os.path.exists(prefix_path):
            raise ValueError(f"Audio file {prefix_path} not found")
        waveform, sample_rate = torchaudio.load(prefix_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        question_audio = waveform.squeeze()

        suffix_audios = []
        for path in suffix_paths:
            audio_path = os.path.join(self.data_dir, path)
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file {audio_path} not found")
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            suffix_audios.append(waveform.squeeze())

        for suffix_audio in suffix_audios:
            complete_audio = torch.cat([question_audio, suffix_audio], dim=-1)
            audio_group.append({
                'array': complete_audio.numpy(),
                'sampling_rate': sample_rate,
            })
        
        return audio_group
    
    def load_data(self):
        logger.info("Preparing data ...")
        dataset = []
        meta_path = os.path.join(self.data_dir, "zh_storyCloze_structured.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file {meta_path} not found.")
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
        for story_meta in tqdm(meta_data):
            idx = story_meta['idx']
            
            prefix_path = os.path.join(story_meta['idx'], story_meta['prefix']['wav'])
            correct_suffix_path = os.path.join(story_meta['idx'], story_meta['correct_suffix']['wav'])
            sSC_suffix_path = os.path.join(story_meta['idx'], story_meta[f'sSC_suffix']['wav'])
            tSC_suffix_path = os.path.join(story_meta['idx'], story_meta[f'tSC_suffix']['wav'])
            
            s_group = self.concat_audio(prefix_path, [correct_suffix_path, sSC_suffix_path])
            t_group = self.concat_audio(prefix_path, [correct_suffix_path, tSC_suffix_path])
            
            story_data = {'idx': idx, 'correct': s_group[0], 'sSC': s_group[1], 'tSC': t_group[1]}
            dataset.append(story_data)
        return dataset
    
    def generate(self, model):
        logger.info("Generating results ...")
        results = []
        for story_item in tqdm(self.dataset, total=len(self.dataset)):
            idx = story_item['idx']
            try:
                correct = story_item['correct']
                sSC = story_item['sSC']
                tSC = story_item['tSC']
                correct_ppl = model.get_ppl(correct, input_type='audio')
                sSC_ppl = model.get_ppl(sSC, input_type='audio')
                tSC_ppl = model.get_ppl(tSC, input_type='audio')
                logger.info(f"Generated ppl for idx {idx}: correct-{correct_ppl} sSC-{sSC_ppl} tSC-{tSC_ppl}")
                logger.info('====================================')
                s_ppl = [correct_ppl, sSC_ppl]
                t_ppl = [correct_ppl, tSC_ppl]
                tmp = {'idx': idx, 's_ppl': s_ppl, 't_ppl': t_ppl}
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        return results
    
    def evaluate(self, results):
        logger.info("Evaluating results ...")
        s_correct = 0
        t_correct = 0
        for story_item in tqdm(results):
            s_answer = np.argmin(story_item['s_ppl'])
            s_correct += (s_answer == 0)
            t_answer = np.argmin(story_item['t_ppl'])
            t_correct += (t_answer == 0)
        s_acc = s_correct / len(results)
        t_acc = t_correct / len(results)
        logger.info("Evaluation completed.")
        return {'s_acc': s_acc, 't_acc': t_acc}
    
    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Generated results saved to {output_file}.")
    
    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        evaluated_result = self.evaluate(generated_results)
        return evaluated_result