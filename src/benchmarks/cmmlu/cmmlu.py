from loguru import logger
from tqdm import tqdm
import json
import os
import pandas as pd
from ..base import BaseBenchmark
import torchaudio
import torch


class CMMLU(BaseBenchmark):
    def __init__(self, data_dir="datas/cmmlu-minimax", **kwargs):
        self.name = 'cmmlu'
        self.data_dir = data_dir
        self.dataset = self.load_data()

    def concat_audio(self, question_paths, choice_paths, splits, pos) -> list:
        audio_group = []

        question_audios = []
        for path in question_paths:
            audio_path = os.path.join(self.data_dir, path)
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file {audio_path} not found, skipping...")
                continue
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                question_audios.append(waveform.numpy().squeeze())
            except Exception as e:
                logger.error(f"Error loading audio file {audio_path}: {str(e)}")
                continue

        choice_audios = []
        for path in choice_paths:
            audio_path = os.path.join(self.data_dir, path)
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file {audio_path} not found, skipping...")
                continue
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                choice_audios.append(waveform.numpy().squeeze())
            except Exception as e:
                logger.error(f"Error loading audio file {audio_path}: {str(e)}")
                continue

        if splits == 1:
            if pos == [0]:
                audio_group = [torch.cat([choice_audio, question_audios[0]]) for choice_audio in choice_audios]
            elif pos == [1]:
                audio_group = [torch.cat([question_audios[0], choice_audio]) for choice_audio in choice_audios]
        elif splits == 2:
            if pos == [1]:
                audio_group = [torch.cat([question_audios[0], choice_audio, question_audios[1]]) for choice_audio in choice_audios]
        else:
            raise ValueError(f"Unsupported splits {splits} with pos {pos}")
        
        return audio_group

    def load_data(self) -> list:
        dataset = []
        meta_path = os.path.join(self.data_dir, "meta_data.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file {meta_path} not found.")
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            for subject in meta_data:
                data = {'subject': subject['subject'], 'qa': []}
                for idx, qa in enumerate(subject['qa']):
                    audio_group = self.concat_audio(qa['question'], qa['choice'], qa['splits'], qa['pos'])
                    right_answer = qa['right_answer']
                    data['qa'].append({'audio_group': audio_group, 'right_answer': right_answer})
                dataset.append(data)
        return dataset

    def generate(self, model):
        logger.add(f'log/{self.name}.log', rotation='50 MB')
        results = []
        for subject_item in tqdm(self.dataset):
            tmp = {'subject': subject_item['subject'], 'response': []}
            logger.info(f"Processing subject: {subject_item['subject']}")
            qa = subject_item['qa']
            try:
                for idx, qa_item in enumerate(qa):
                    audio_group = qa_item['audio_group']
                    right_answer = qa_item['right_answer']
                    logprobs = [model.generate(audio)[1] for audio in audio_group]
                    logger.info(f"Generated logprobs for audio group {idx}: {logprobs}")
                    tmp['response'].append({'idx': idx, 'logprob': logprobs, 'right_answer': right_answer})
                logger.info('====================================')
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        return results

    def evaluate(self, results):
        pass

    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        logger.info("Generated results saved successfully.")
        return ''