from loguru import logger
from tqdm import tqdm
import json
import os
import pandas as pd
from ..base import BaseBenchmark
from .evaluate import _evaluate
import torchaudio


class OpenAudioBench(BaseBenchmark):
    def __init__(self, subset_name, data_dir="datas/openaudiobench", **kwargs):
        self.name = 'openaudiobench'
        self.subset_name = subset_name
        self.data_dir = data_dir
        self.dataset = self.load_data()
        
    def load_data(self):
        data_csv_path = os.path.join(self.data_dir, self.subset_name, f"{self.subset_name}.csv")
        if not os.path.exists(data_csv_path):
            raise FileNotFoundError(f"Data file {data_csv_path} not found.")
        
        df = pd.read_csv(data_csv_path)
        dataset = []
        
        for _, row in df.iterrows():
            data = row.to_dict()
            # 加载音频文件
            audio_path = os.path.join(self.data_dir, self.subset_name, data['audio_path'])
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file {audio_path} not found, skipping...")
                continue
                
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                    
                data['audio'] = {
                    'array': waveform.numpy().squeeze(),
                    'sampling_rate': sample_rate
                }
                dataset.append(data)
            except Exception as e:
                logger.error(f"Error loading audio file {audio_path}: {str(e)}")
                continue
                
        return dataset
    
    def generate(self, model):
        logger.add(f'log/{self.name}-{self.subset_name}.log', rotation='50MB')

        results = []
        for item in tqdm(self.dataset, total=len(self.dataset)):
            tmp = {k: v for k, v in item.items() if k != 'audio'}
            logger.info(item['instruction'])
            try:
                response = model.generate_audio(item['audio'])
                logger.info(response)
                logger.info('====================================')
                tmp['infer_response'] = response
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        
        if len(self.dataset) > len(results):
            logger.warning(f"Some data failed to process. {len(self.dataset) - len(results)} items were skipped.")
        
        return results

    def evaluate(self, data):
        evaluated_results = _evaluate(data, self.subset_name)
        logger.info("Evaluation completed.")
        return evaluated_results
    

    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-{self.subset_name}.jsonl')
        with open(output_file, 'w') as f:
            for record in results:
                json_line = json.dumps(record)
                f.write(json_line + '\n')
        logger.info(f"Generated results saved to {output_file}.")


    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        evaluated_results = self.evaluate(generated_results)
        logger.info("Run completed.")
        return evaluated_results
