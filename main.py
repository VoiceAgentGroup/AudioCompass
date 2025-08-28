import os
from loguru import logger
from argparse import ArgumentParser
from src.models import load_model, list_models
from src.benchmarks import load_benchmark, list_benchmarks


def main():
    parser = ArgumentParser()
    parser.add_argument('--model-name', type=str, default='repulsar_sft', choices=list_models())
    parser.add_argument('--ckpt', type=str, default="/inspire/hdd/project/embodied-multimodality/public/zxu/megatron_workspace/eval/ckpts/RepulsarSFT-v0.3.2-1/04000")
    parser.add_argument('--benchmark', type=str, default='mmlu', choices=list_benchmarks())
    parser.add_argument('--subset', type=str, default='mmsu')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--timbre', type=str, default='echo')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--offline', action="store_true")
    args = parser.parse_args()
    
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'

    # load benchmark
    logger.info("Loading benchmark ...")
    benchmark = load_benchmark(**vars(args))
    
    # load model
    logger.info(f"Loading model {args.model_name} ...")
    model = load_model(**vars(args))

    # generate results
    result = benchmark.run(model, args.output_dir)

    if result is not None:
        logger.info(result)


if __name__ == '__main__':
    main()
