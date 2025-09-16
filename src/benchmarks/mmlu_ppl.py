from loguru import logger
from tqdm import tqdm
import json
import os
import numpy as np
from .base import BaseBenchmark
import torchaudio
import torch

class MMLU(BaseBenchmark):
    def __init__(self, data_dir="datas/mmlu", cache_dir='cache', **kwargs):
        self.name = 'mmlu_ppl'
        self.data_dir = os.path.join(cache_dir, data_dir)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.add(f'log/{self.name}-{timestamp}.log', rotation='50MB')
        self.dataset = self.load_data()

    def concat_audio(self, question_path, choice_path) -> list:
        question_path = os.path.join(self.data_dir, question_path)
        if not os.path.exists(question_path):
            raise ValueError(f"Audio file {question_path} not found")
        question_audio, sample_rate = torchaudio.load(question_path)
        if question_audio.shape[0] > 1:
            question_audio = question_audio.mean(dim=0, keepdim=True)
            
        choice_path = os.path.join(self.data_dir, choice_path)
        if not os.path.exists(choice_path):
            raise ValueError(f"Audio file {choice_path} not found")
        choice_audio, sample_rate = torchaudio.load(choice_path)
        if choice_audio.shape[0] > 1:
            choice_audio = choice_audio.mean(dim=0, keepdim=True)

        complete_audio = torch.cat([question_audio, choice_audio], dim=-1)
        
        return complete_audio, sample_rate
    
    def concat_with_silence(self, audios, sample_rate):
        silence = torch.zeros((audios[0].shape[0], int(0.5 * sample_rate)))  # 0.5 second silence
        combined = audios[0]
        for segment in audios[1:]:
            combined = torch.cat((combined, silence, segment), dim=1)
        return {
            "array": combined.squeeze(0).numpy(),
            "sampling_rate": sample_rate
        }

    def load_data(self) -> list:
        logger.info("Preparing data ...")
        dataset = []
        meta_path = os.path.join(self.data_dir, "meta.json")
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        for subject, subject_item in tqdm(meta_data.items(), desc="Loading subjects"):
            data_item = {'subject': subject, 'qa': []}
            
            if 'prompt' not in subject_item:
                continue
            prompt_path = subject_item['prompt']['path']
            prompt_audio, _ = torchaudio.load(os.path.join(self.data_dir, prompt_path))
            if prompt_audio.shape[0] > 1:
                prompt_audio = prompt_audio.mean(dim=0, keepdim=True)
                
            # prepare test audios
            for test_qa in subject_item['qa']:
                audio_group = []
                question_path = test_qa['question']['path']
                for choice in test_qa['choices']:
                    choice_path = choice['path']
                    test_audio, sample_rate = self.concat_audio(question_path, choice_path)
                    audio = self.concat_with_silence([prompt_audio, test_audio], sample_rate)
                    audio_group.append(audio)
                right_answer = test_qa['right_answer']
                data_item['qa'].append({'audio_group': audio_group, 'right_answer': right_answer})
            dataset.append(data_item)
        return dataset

    def generate(self, model):
        logger.info("Generating results ...")
        results = []
        for subject_item in tqdm(self.dataset, total=len(self.dataset)):
            tmp = {'subject': subject_item['subject'], 'response': []}
            logger.info(f"Processing subject: {subject_item['subject']}")
            qa = subject_item['qa']
            try:
                for idx, qa_item in enumerate(qa):
                    audio_group = qa_item['audio_group']
                    right_answer = qa_item['right_answer']
                    ppl = [model.get_ppl(audio, input_type='audio') for audio in audio_group]
                    logger.info(f"Generated ppl for audio group {idx}: {ppl}")
                    tmp['result'].append({'idx': idx, 'ppl': ppl, 'right_answer': right_answer})
                logger.info('====================================')
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        return results

    def evaluate(self, results):
        logger.info("Evaluating results ...")
        choice_strs = ['A', 'B', 'C', 'D']
        correct = 0
        total = 0
        for subject_item in tqdm(results):
            logger.info("Subject: " + subject_item['subject'])
            subject_results = subject_item['result']
            for result in subject_results:
                answer = choice_strs[np.argmax(result['ppl'])]
                correct += (answer == result['right_answer'])
                logger.info(f"idx: {result['idx']} answer: {answer} right_answer: {result['right_answer']}")
                total += 1
        acc = correct / total
        logger.info("Evaluation completed.")
        return {'acc': acc}
    
    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Generated results saved to {output_file}.")
    
    def run(self, model, output_dir):
        return super().run(model, output_dir)