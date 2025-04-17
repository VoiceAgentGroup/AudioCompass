import os
from loguru import logger
from argparse import ArgumentParser
from src.models import load_model, list_models
from src.benchmarks import load_benchmark, list_benchmarks


def main():
    parser = ArgumentParser()
    parser.add_argument('--model-name', type=str, default='speechgpt2', choices=list_models())
    parser.add_argument('--benchmark', type=str, default='voicebench', choices=list_benchmarks())
    parser.add_argument('--subset', type=str, default='alpacaeval')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--offline', action="store_true")
    args = parser.parse_args()
    
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'

    # load benchmark
    benchmark = load_benchmark(benchmark_name=args.benchmark, subset_name=args.subset, split=args.split, cache_dir=args.cache_dir, offline=args.offline)
    
    # load model
    model = load_model(args.model_name)

    # generate results
    result = benchmark.run(model, args.output_dir)

    if result is not None:
        logger.info(result)


if __name__ == '__main__':
    main()
