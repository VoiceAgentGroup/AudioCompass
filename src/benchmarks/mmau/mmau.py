from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
import json
import os
from .evaluate import _evaluate
from ..base import BaseBenchmark


class MMAU(BaseBenchmark):
    def __init__(self, split, **kargs):
        self.name = 'mmau'
        self.split = split
        self.dataset = self.load_data()

    def load_data(self):
        dataset = load_dataset('lmms-lab/mmau', split=self.split)
        return dataset
    
    def generate(self, model):
        logger.add(f'log/{self.name}-{self.split}.log', rotation='50MB')

        results = []
        output_keys = ['question_id', 'question', 'choices', 'answer', 'task', 'sub-category', 'difficulty']
        for item in tqdm(self.dataset, total=len(self.dataset)):
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
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        if len(self.dataset) > len(results):
            logger.warning(f"Some data failed to process. {len(self.dataset) - len(results)} items were skipped.")

        return results
    

    def evaluate(self, data):
        evaluated_results = _evaluate(data)
        logger.info("Evaluation completed.")
        return evaluated_results
    

    def save_generated_results(self, results, output_dir, model_name):
        os.makedirs(output_dir, exist_ok=True)
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