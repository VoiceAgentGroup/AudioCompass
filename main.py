from datasets import load_dataset, Audio
from argparse import ArgumentParser
from src.models import model_cls_mapping
import json
from tqdm import tqdm
from loguru import logger
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='local', choices=list(model_cls_mapping.keys()))
    parser.add_argument('--data', type=str, default='alpacaeval')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'text', 'ttft'])
    args = parser.parse_args()

    # load data
    data = load_dataset('hlt-lab/voicebench', args.data, split=args.split)
    data = data.cast_column("audio", Audio(sampling_rate=16_000))

    # load model
    model = model_cls_mapping[args.model]()
    # data = data.select([0,1,2,3,4,5])

    if args.modality == 'ttft':
        # avoid cold start
        _ = model.generate_ttft(data[0]['audio'])

    # inference
    logger.add('log/last.log', rotation='50MB')
    results = []
    for item in tqdm(data, total=len(data)):
        tmp = {k: v for k, v in item.items() if k != 'audio'}
        try:
            if args.modality == 'text':
                response = model.generate_text(item['prompt'])
                token_usage = None
            elif args.modality == 'audio':
                mixed_response = model.generate_audio(item['audio'])
                if isinstance(mixed_response, tuple):
                    response, token_usage = mixed_response
                else:
                    response = mixed_response
                    token_usage = None
            elif args.modality == 'ttft':
                response = model.generate_ttft(item['audio'])
                token_usage = None
            else:
                raise NotImplementedError
            logger.info(item['prompt'])
            logger.info(response)
            logger.info('====================================')
            tmp['response'] = response
            if token_usage:
                tmp['token_usage'] = token_usage
            results.append(tmp)
        except Exception as e:
            logger.error(f"Error with prompt: {item['prompt']}")
            logger.error(e)
            logger.error('====================================')
            continue

    if len(data) > len(results):
        logger.warning(f"Some data failed to process. {len(data) - len(results)} items were skipped.")

    # save results
    model_name = args.model.split('/')[-1]
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{model_name}-{args.data}-{args.split}-{args.modality}.jsonl')
    with open(output_file, 'w') as f:
        for record in results:
            json_line = json.dumps(record)  # Convert dictionary to JSON string
            f.write(json_line + '\n')


if __name__ == '__main__':
    main()
