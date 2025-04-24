from datasets import load_dataset, Audio
from loguru import logger
from tqdm import tqdm
import json
import os
from .evaluate import _evaluate
from ..base import BaseBenchmark


class VoiceBench(BaseBenchmark):
    def __init__(self, subset_name, split, data_dir='datas/voicebench', cache_dir='cache', **kwargs):
        self.name = 'voicebench'
        self.subset_name = subset_name
        self.split = split
        self.data_dir = os.path.join(cache_dir, data_dir)
        logger.add(f'log/{self.name}-{self.subset_name}-{self.split}.log', rotation='50MB')
        self.dataset = self.load_data()

    
    def load_data(self):
        logger.info("Preparing data ...")
        # if kwargs.get('offline', None) == True:
        #     dataset = load_dataset('parquet', data_dir=self.data_dir, trust_remote_code=True)
        #     dataset = dataset[self.subset_name][self.split]
        # else:
        #     dataset = load_dataset('hlt-lab/voicebench', self.subset_name, split=self.split, cache_dir=self.data_dir)
        dataset = load_dataset('hlt-lab/voicebench', self.subset_name, split=self.split, cache_dir=self.data_dir)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        return dataset
    
    
    def generate(self, model):
        logger.info("Generating results ...")

        results = []
        for item in tqdm(self.dataset, total=len(self.dataset)):
            tmp = {k: v for k, v in item.items() if k != 'audio'}
            logger.info(item['prompt'])
            try:
                response = model.generate_a2t(item['audio'])
                logger.info(response)
                logger.info('====================================')
                tmp['response'] = response
                results.append(tmp)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        if len(self.dataset) > len(results):
            logger.warning(f"Some data failed to process. {len(self.dataset) - len(results)} items were skipped.")

        return results
    
    
    def evaluate(self, data):
        dataset_evaluator_mapping = {
            'alpacaeval': 'open',
            'commoneval': 'open',
            'sd-qa': 'qa',
            'ifeval': 'ifeval',
            'advbench': 'harm',
            'openbookqa': 'mcq',
            'mmsu': 'mcq',
        }
        evaluated_results = _evaluate(data, dataset_evaluator_mapping[self.subset_name])
        logger.info("Evaluation completed.")
        return evaluated_results
    

    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_name.split('/')[-1]
        output_file = os.path.join(output_dir, f'{model_name}-{self.name}-{self.subset_name}-{self.split}.jsonl')
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