from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
import json
import os
import gc
from .evaluate import _evaluate
from ..base import BaseBenchmark


class MMAU(BaseBenchmark):
    def __init__(self, split, data_dir='datas/mmau', cache_dir='cache', **kwargs):
        self.name = 'mmau'
        self.split = split
        self.data_dir = os.path.join(cache_dir, data_dir)
        logger.add(f'log/{self.name}-{self.split}.log', rotation='50MB')
        self.dataset = self.load_data(**kwargs)
        
        
    def check_split(self, split):
        available_split = ['test', 'test-mini']
        if split not in available_split:
            raise ValueError("Split should be one of " + available_split)
        
        
    def load_data(self, **kwargs):
        logger.info("Preparing data ...")
        if kwargs.get('offline', None) == True:
            dataset = load_dataset('parquet', data_dir=self.data_dir, trust_remote_code=True)
            dataset = dataset[self.split]
        else:
            dataset = load_dataset('lmms-lab/mmau', split=self.split, **kwargs)
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
    
    
    def get_result_template(self):
        template_file = os.path.join(self.data_dir, 'mmau-test.json')
        with open(template_file, 'r') as f:
            template = json.load(f)
        return template
    

    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-{self.split}.json')
        with open(output_file, 'w') as f:
            if self.split == 'test_mini':
                json.dump(results, f, indent=4)
            else:
                template = self.get_result_template()
                if len(template) != len(results):
                    raise RuntimeError("Fail to apply submission template due to some missed results.")
                for template_item, result_item in zip(template, results):
                    template_item['model_prediction'] = result_item['model_prediction']
                json.dump(template, f, indent=4)
                
        logger.info(f"Generated results saved to {output_file}.")


    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        if self.split == 'test_mini':
            evaluated_results = self.evaluate(generated_results)
        else:
            evaluated_results = None
        logger.info("Run completed.")
        return evaluated_results