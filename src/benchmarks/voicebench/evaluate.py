from argparse import ArgumentParser
import json
from .evaluator import evaluator_mapping
from loguru import logger


def _evaluate(data, evaluator_name):
    evaluator = evaluator_mapping[evaluator_name]()
    results = evaluator.evaluate(data)
    if not results:
        raise ValueError("No results returned from evaluator.")
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--src-file', type=str, required=True)
    parser.add_argument('--evaluator', type=str, required=True, choices=list(evaluator_mapping.keys()))
    args = parser.parse_args()
    data = []
    with open(args.src_file, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())  # Convert JSON string to dictionary
            data.append(json_obj)
    results = _evaluate(data, args.evaluator)
    logger.info(results)
