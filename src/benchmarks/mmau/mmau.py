from datasets import load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm
import json
import os
import gc
from .evaluate import _evaluate
from ..base import BaseBenchmark


class MMAU(BaseBenchmark):
    def __init__(self, split, batch_size=500, **kwargs):
        self.name = 'mmau'
        self.split = split
        self.batch_size = batch_size
        self.dataset = self.load_data(**kwargs)
        
    def load_data(self, **kwargs):
        logger.info("Preparing data ...")
        if kwargs.get('offline', None) == True:
            dataset = load_dataset('parquet', data_dir='datas/mmau', trust_remote_code=True)
            dataset = dataset[self.split]
        else:
            dataset = load_dataset('lmms-lab/mmau', split=self.split, **kwargs)
        return dataset
    
    def generate(self, model):
        logger.info("Generating results ...")
        logger.add(f'log/{self.name}-{self.split}.log', rotation='50MB')
        
        results = []
        output_keys = ['question_id', 'question', 'choices', 'answer', 'task', 'sub-category', 'difficulty']
        
        batch = []
        count = 0
        total_processed = 0
        
        for item in self.dataset:
            batch.append(item)
            count += 1
            
            if count >= self.batch_size:
                self._process_batch(batch, model, results, output_keys)
                total_processed += len(batch)
                
                batch = []
                count = 0
                gc.collect()
        
        if batch:
            self._process_batch(batch, model, results, output_keys)
            total_processed += len(batch)
        
        if total_processed > len(results):
            logger.warning(f"Some data failed to process. {total_processed - len(results)} items were skipped.")
        
        return results
    
    def _process_batch(self, batch, model, results, output_keys):
        for item in tqdm(batch, total=len(batch)):
            tmp = {k: v for k, v in item.items() if k in output_keys}
            input_text = item['question'] + ' ' + item['choices']
            input_audio = item['audio']
            logger.info(input_text)
            try:
                response = model.generate_mixed(input_audio, input_text)
                logger.info(response)
                logger.info('====================================')
                tmp['response'] = response
                results.append(tmp)
                
                del input_audio
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue

    def evaluate(self, data):
        evaluated_results = _evaluate(data)
        logger.info("Evaluation completed.")
        return evaluated_results
    

    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-{self.split}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Generated results saved to {output_file}.")


    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.name)
        evaluated_results = self.evaluate(generated_results)
        logger.info("Run completed.")
        return evaluated_results