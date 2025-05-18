from datasets import load_dataset, Audio
from huggingface_hub import snapshot_download
from loguru import logger
from tqdm import tqdm
import json
import os
import gc
import tarfile
from .evaluate import _evaluate
from ..base import BaseBenchmark


class MMAR(BaseBenchmark):
    def __init__(self, data_dir='datas/MMAR', cache_dir='cache', **kwargs):
        self.name = 'mmar'
        self.data_dir = os.path.join(cache_dir, data_dir)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.add(f'log/{self.name}-{timestamp}.log', rotation='50MB')
        
        self.dataset = self.load_data(**kwargs)
        
        
    def load_data(self, **kwargs):
        logger.info("Preparing data ...")
        if kwargs.get('offline', None) == None:
            snapshot_download(repo_id='BoJack/MMAR', repo_type='dataset', local_dir=self.data_dir)
        if not os.path.exists(os.path.join(self.data_dir, 'audio')):
            tf = tarfile.open(os.path.join(self.data_dir, 'mmar-audio.tar.gz'), 'r:gz')
            tf.extractall(self.data_dir)
        dataset = load_dataset(self.data_dir, trust_remote_code=True)['test']
        
        def add_path_prefix(x):
            x['audio_path'] = os.path.join(self.data_dir, x['audio_path'])
            return x
        dataset = dataset.map(add_path_prefix)
        dataset = dataset.cast_column('audio_path', Audio()).rename_column('audio_path', 'audio')
        return dataset
    
    
    def generate(self, model):
        logger.info("Generating results ...")
        
        results = []
        output_keys = ['question', 'choices', 'answer', 'task', 'sub-category', 'difficulty']
        
        for item in tqdm(self.dataset):
            tmp = {k: v for k, v in item.items() if k in output_keys}
            input_text = item['question'] + ' ' + item['choices']
            input_audio = item['audio']
            logger.info(input_text)
            try:
                response = model.generate_at2t(input_audio, input_text)
                logger.info(response)
                logger.info('====================================')
                tmp['model_prediction'] = response
                results.append(tmp)
                
                del input_audio
                gc.collect()
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
                
        return results
    
    
    def evaluate(self, data):
        evaluated_results = _evaluate(data)
        logger.info("Evaluation completed.")
        return evaluated_results
    

    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Generated results saved to {output_file}.")


    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        evaluated_results = self.evaluate(generated_results)
        logger.info("Run completed.")
        return evaluated_results