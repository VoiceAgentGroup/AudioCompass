from benchmarks import load_benchmark, Audio
from loguru import logger
from tqdm import tqdm
import json
import os
from evaluate import _evaluate
from base import BaseBenchmark


class VoiceBench(BaseBenchmark):
    def __init__(self, subset_name, split):
        self.name = 'voicebench'
        self.subset_name = subset_name
        self.split = split
        self.dataset = self.load_data()

    
    def load_data(self):
        dataset = load_benchmark('hlt-lab/voicebench', self.subset_name, split=self.split)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        return dataset
    
    
    def generate(self, model, output_dir):
        logger.add(f'log/{self.name}-{self.subset_name}-{self.split}.log', rotation='50MB')

        results = []
        for item in tqdm(self.dataset, total=len(self.dataset)):
            tmp = {k: v for k, v in item.items() if k != 'audio'}
            logger.info(item['prompt'])
            try:
                response = model.generate(item['audio'])
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

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{model.model_name}-{self.name}-{self.subset_name}-{self.split}.jsonl')
        with open(output_file, 'w') as f:
            for record in results:
                json_line = json.dumps(record)
                f.write(json_line + '\n')
        logger.info(f"Results saved to {output_file}.")

        return results
    
    
    def evaluate(self, data):
        evaluated_results = _evaluate(data)
        logger.info("Evaluation completed.")
        return evaluated_results


    def run(self, model, output_dir):
        results = self.generate(model, output_dir)
        evaluated_results = self.evaluate(results)
        return evaluated_results