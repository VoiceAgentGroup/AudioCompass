from loguru import logger
from tqdm import tqdm
import json
import os
from .base import BaseBenchmark
from src.transcriptors import WhisperLargeV3
from src.utils.extractor import Extractor
import torchaudio
import torch
import datetime

class CMMLU(BaseBenchmark):
    def __init__(self, data_dir="datas/cmmlu", cache_dir='cache', **kwargs):
        self.name = 'cmmlu_gen'
        self.data_dir = os.path.join(cache_dir, data_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.add(f'log/{self.name}-{timestamp}.log', rotation='5MB')
        self.transcriptor = WhisperLargeV3(**kwargs)
        self.dataset = self.load_data()
    
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
                question_path = test_qa['question']['path']
                question_audio, sample_rate = torchaudio.load(os.path.join(self.data_dir, question_path))
                audio = self.concat_with_silence([prompt_audio, question_audio], sample_rate)
                right_answer = test_qa['right_answer']
                
                question_text = test_qa['question']['text']
                choice_text = ' '.join([t['text'] for t in test_qa['choices']])
                text = f'{question_text} {choice_text}'
                data_item['qa'].append({'question': text, 'audio': audio, 'right_answer': right_answer})
            dataset.append(data_item)
        return dataset

    def generate(self, model):
        logger.info("Generating results ...")
        results = []
        for subject_item in tqdm(self.dataset, total=len(self.dataset)):
            tmp = {'subject': subject_item['subject'], 'result': []}
            logger.info(f"Processing subject: {subject_item['subject']}")
            qa = subject_item['qa']
            try:
                for idx, qa_item in enumerate(qa):
                    logger.info(f"Question: {qa_item['question']}")
                    audio = qa_item['audio']
                    right_answer = qa_item['right_answer']
                    response_audio, sample_rate = model.generate_a2a(audio)
                    transcription = self.transcriptor.inference(response_audio, generate_kwargs={"language": "english"})
                    logger.info(f"Response: {transcription}")
                    tmp['result'].append({'idx': idx, 'response': transcription, 'right_answer': right_answer})
                logger.info('====================================')
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        return results

    def evaluate(self, results):
        logger.info("Evaluating results ...")
        extractor = Extractor()
        correct = 0
        total = 0
        for subject_item in tqdm(results):
            logger.info("Subject: " + subject_item['subject'])
            subject_results = subject_item['result']
            for result in subject_results:
                answer = extractor.rule_extract(result['response'])
                if answer is not None and answer.lower() == result['right_answer'].lower():
                    correct += 1
                logger.info(f"idx: {result['idx']} answer: {answer} right_answer: {result['right_answer']}")
                total += 1
        acc = correct / total
        logger.info("Evaluation completed.")
        return {'acc': acc}
    
    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-{timestamp}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Generated results saved to {output_file}.")
    
    def run(self, model, output_dir):
        return super().run(model, output_dir)