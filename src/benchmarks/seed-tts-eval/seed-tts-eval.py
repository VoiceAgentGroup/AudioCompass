import os
import torch
import torchaudio
import string
from tqdm import tqdm
from loguru import logger
from zhon.hanzi import punctuation
from jiwer import compute_measures
import json
from ..base import BaseBenchmark
from src.transcriptors import Paraformer

punctuation_all = punctuation + string.punctuation

class SeedTTSEval(BaseBenchmark):
    def __init__(self, split, data_dir="datas/seedtts_testset", **kwargs):
        self.name = 'seed-tts-eval'
        self.split = split
        self.data_dir = data_dir
        self.dataset = self.load_data()
        self.transcriptor = Paraformer(**kwargs)
        logger.add(f'log/{self.name}', rotation='50 MB')
        
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
            meta_path = os.path.join(self.data_dir, 'zh/meta.lst')
            wav_dir = os.path.join(self.data_dir, 'zh/wavs')
        with open(meta_path, 'r') as f:
            lines = f.readline()
        
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
        
        for idx, data in enumerate(tqdm(self.dataset)):
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
    
    def process_one_wer(self, hypo, truth):
        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            if x == '\'':
                continue
            truth = truth.replace(x, '')
            hypo = hypo.replace(x, '')

        truth = truth.replace('  ', ' ')
        hypo = hypo.replace('  ', ' ')

        if self.split == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif self.split == "en":
            truth = truth.lower()
            hypo = hypo.lower()
        else:
            raise NotImplementedError

        measures = compute_measures(truth, hypo)
        ref_list = truth.split(" ")
        wer = measures["wer"]
        subs = measures["substitutions"] / len(ref_list)
        dele = measures["deletions"] / len(ref_list)
        inse = measures["insertions"] / len(ref_list)
        return (raw_truth, raw_hypo, wer, subs, dele, inse)
    
    def evaluate(self, results):
        logger.info("Evaluating results ...")
        # wer
        total_wer = 0
        for result in tqdm(results):
            raw_truth, raw_hypo, wer, subs, dele, inse = self.process_one_wer(result['transcription'], result['infer_text'])
            total_wer += wer
        avg_wer = total_wer / len(results)
        return {'wer': avg_wer}
    
    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, self.name)
        wav_dir = os.path.join(output_dir, 'tts-wavs')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        for result in results:
            wav_path = os.path.join(wav_dir, f"{model_name}-{result['idx']}.wav")
            wav = result['tts_wav'] if result['tts_wav'].ndim == 2 else result['tts_wav'].unsqueeze(0)
            torchaudio.save(wav_path, wav, result['sample_rate'])
            result['tts_wav_path'] = wav_path
            result.pop('tts_wav')
            result.pop('sample_rate')
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