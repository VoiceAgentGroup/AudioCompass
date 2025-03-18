from datasets import load_dataset, Audio
from loguru import logger
from tqdm import tqdm

class VoiceBench:
    def __init__(self, subset_name, split):
        self.name = 'voicebench'
        self.subset_name = subset_name
        self.split = split
        self.dataset = self.load_data()
    
    def load_data(self):
        dataset = load_dataset('hlt-lab/voicebench', self.subset_name, split=self.split)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        return dataset
    
    def generate(self, model):
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

        return results  