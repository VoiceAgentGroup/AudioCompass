from loguru import logger
from tqdm import tqdm
import json
import os
import numpy as np
from .base import BaseBenchmark
from utils.rule_extractor import extract_answer
import torchaudio
import torch


class CMMLU(BaseBenchmark):
    def __init__(self, data_dir="datas/cmmlu-minimax", **kwargs):
        self.name = 'cmmlu'
        self.data_dir = data_dir
        self.dataset = self.load_data()

    def concat_audio(self, question_path, choice_path) -> list:
        audio_group = []

        question_path = os.path.join(self.data_dir, question_path)
        if not os.path.exists(question_path):
            raise ValueError(f"Audio file {question_path} not found")
        waveform, sample_rate = torchaudio.load(question_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        question_audio = waveform.squeeze()

        choice_audios = []
        for path in choice_path:
            audio_path = os.path.join(self.data_dir, path)
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file {audio_path} not found")
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            choice_audios.append(waveform.squeeze())

        for choice_audio in choice_audios:
            complete_audio = torch.cat([question_audio, choice_audio], dim=-1)
            audio_group.append({
                'array': complete_audio.numpy(),
                'sampling_rate': sample_rate,
            })
            
        question_audio = {
            'array': question_audio,
            'sampling_rate': sample_rate,
        }
        
        return audio_group, question_audio

    def load_data(self) -> list:
        logger.info("Preparing data ...")
        dataset = []
        meta_path = os.path.join(self.data_dir, "meta_data.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file {meta_path} not found.")
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            for subject in tqdm(meta_data):
                data = {'subject': subject['subject'], 'qa': []}
                for idx, qa in enumerate(subject['qa']):
                    question_path = qa['question']['path']
                    choice_path = [choice['path'] for choice in qa['choice']]
                    audio_group, question_audio = self.concat_audio(question_path, choice_path)
                    
                    choice_lbs = ['A. ', 'B. ', 'C. ', 'D. ']
                    choice_descs = [choice['text'] for choice in qa['choice']]
                    choice_text = ' '.join([(choice_lb + choice_desc) for choice_lb, choice_desc in zip(choice_lbs, choice_descs)])
                    
                    right_answer = qa['right_answer']
                    data['qa'].append({'audio_group': audio_group, 'question_audio': question_audio, 'choice_text': choice_text, 'right_answer': right_answer})
                dataset.append(data)
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
      
    def generate_ppl(self, model):
        logger.info("Generating results ...")
        logger.add(f'log/{self.name}-ppl.log', rotation='50 MB')
        results = []
        for subject_item in tqdm(self.dataset, total=len(self.dataset)):
            tmp = {'subject': subject_item['subject'], 'response': []}
            logger.info(f"Processing subject: {subject_item['subject']}")
            qa = subject_item['qa']
            try:
                for idx, qa_item in enumerate(qa):
                    audio_group = qa_item['audio_group']
                    right_answer = qa_item['right_answer']
                    logprobs = [self.process_logprob(model.generate_audio(audio, max_new_tokens=1)[1]) for audio in audio_group]
                    logger.info(f"Generated logprobs for audio group {idx}: {logprobs}")
                    tmp['response'].append({'idx': idx, 'logprob': logprobs, 'right_answer': right_answer})
                logger.info('====================================')
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        return results

    def evaluate_ppl(self, results):
        logger.info("Evaluating results ...")
        choice_strs = ['A', 'B', 'C', 'D']
        correct = 0
        total = 0
        for subject_item in tqdm(results):
            subject_results = subject_item['response']
            for result in subject_results:
                answer = choice_strs[np.argmax(result['logprob'])]
                correct += (answer == result['right_answer'])
                total += 1
        acc = correct / total
        logger.info("Evaluation completed.")
        return {'acc': acc}
    
    def save_generated_result_ppl(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-ppl.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Generated results saved to {output_file}.")
        
    
    def generate_normal(self, model):
        logger.info("Generating results ...")
        logger.add(f'log/{self.name}-normal.log', rotation='50 MB')
        prompt = 'Here is an audio of a question. Please listen carefully and choose the right answer.\n'
        results = []
        for subject_item in tqdm(self.dataset, total=len(self.dataset)):
            tmp = {'subject': subject_item['subject'], 'response': []}
            logger.info(f"Processing subject: {subject_item['subject']}")
            qa = subject_item['qa']
            try:
                for idx, qa_item in enumerate(qa):
                    question = qa_item['question_audio']
                    choice_text = qa_item['choice_text']
                    right_answer = qa_item['right_answer']
                    answer, _ = model.generate_mixed(question, prompt + choice_text)
                    logger.info(f"Generated response for audio group {idx}: {answer}")
                    tmp['response'].append({'idx': idx, 'answer': answer, 'right_answer': right_answer})
                logger.info('====================================')
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        return results
    
    def evaluate_normal(self, results):
        correct = 0
        total = 0
        for subject_item in tqdm(results):
            subject_results = subject_item['response']
            for result in subject_results:
                answer = extract_answer(result['answer'])
                correct += (answer == result['right_answer'])
                total += 1
        acc = correct / total
        logger.info("Evaluation completed.")
        return {'acc': acc}
    
    def save_generated_results_normal(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-normal.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Generated results saved to {output_file}.")
        
    
    def generate(self, model):
        return self.generate_ppl(model)
    
    def evaluate(self, results):
        return self.evaluate_ppl(results)
    
    def save_generated_results(self, results, output_dir, model_name):
        return self.save_generated_results_normal(results, output_dir, model_name)
    
    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        evaluated_result = self.evaluate(generated_results)
        return evaluated_result