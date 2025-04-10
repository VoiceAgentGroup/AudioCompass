from loguru import logger
from tqdm import tqdm
import json
import os
import numpy as np
from .base import BaseBenchmark
from utils.rule_extractor import extract_answer
import torchaudio
import torch


class StoryCloze(BaseBenchmark):
    def __init__(self, split, data_dir="datas/zh-storycloze", **kwargs):
        self.name = 'storycloze'
        self.check_split(split)
        self.split = split
        self.data_dir = data_dir
        self.dataset = self.load_data()
        
    def check_split(self, split):
        available_split = ['sSC', 'tSC']
        if split not in available_split:
            raise ValueError("Split should be one of", available_split)
        
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

        for choice_audio in suffix_audios:
            complete_audio = torch.cat([question_audio, choice_audio], dim=-1)
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
            prefix_path = os.path.join(self.data_dir, story_meta['idx'], story_meta['prefix'])
            correct_suffix_path = os.path.join(self.data_dir, story_meta['idx'], story_meta['correct_suffix'])
            fake_suffix_path = os.path.join(self.data_dir, story_meta['idx'], story_meta[f'{self.split}_suffix'])
            audio_group = self.concat_audio(prefix_path, [correct_suffix_path, fake_suffix_path])
            story_data = {'idx': idx, 'audio_group': audio_group}
            dataset.append(story_data)
        return dataset
    
    def process_logprob(self, logprob):
        audio_logprob = 0
        length = 0
        for token in logprob[1:]:
            token = list(token.values())[0]
            if token['decoded_token'] == '<|AUDIO|>':
                audio_logprob += token['logprob']
                length += 1      
        audio_logprob /= length
        return audio_logprob
    
    def generate(self, model):
        logger.info("Generating results ...")
        logger.add(f'log/{self.name}-{self.split}.log', rotation='50 MB')
        results = []
        for story_item in tqdm(self.dataset, total=len(self.dataset)):
            idx = story_item['idx']
            try:
                audio_group = story_item['audio_group']
                logprobs = [self.process_logprob(model.generate_audio(audio)[1]) for audio in audio_group]
                logger.info(f"Generated logprobs for audio group {idx}: {logprobs}")
                logger.info('====================================')
                tmp = {'idx': idx, 'logprobs': logprobs}
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        return results
    
    def evaluate(self, results):
        logger.info("Evaluating results ...")
        correct = 0
        for story_item in tqdm(results):
            answer = np.argmax(story_item['logprobs'])
            correct += (answer == 0)
        acc = correct / len(results)
        logger.info("Evaluation completed.")
        return {'acc': acc}
    
    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-{self.split}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Generated results saved to {output_file}.")
    
    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        evaluated_result = self.evaluate(generated_results)
        return evaluated_result