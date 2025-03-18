from argparse import ArgumentParser
from src.models import load_model, list_models
from src.benchmarks import load_benchmark, list_benchmarks
import json
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='local', choices=list_models())
    parser.add_argument('--dataset', type=str, default='voicebench', choices=list_benchmarks())
    parser.add_argument('--subset', type=str, default='alpacaeval')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    # load data
    dataset = load_benchmark(benchmark_name=args.dataset, subset_name=args.subset, split=args.split)

    # load model
    model = load_model(args.model)

    results = dataset.generate(model)

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
