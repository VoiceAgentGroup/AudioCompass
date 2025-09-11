from .base import Evaluator
import numpy as np
import random
import json
from argparse import ArgumentParser
from src.utils.extractor import Extractor

class MCQEvaluator(Evaluator):
    def evaluate(self, data):
        extractor = Extractor()
        ground_truth = [item['reference'] for item in data]
        preds = [extractor.rule_extract(item['response']) for item in data]
        cnt = 0
        for idx in range(len(preds)):
            if preds[idx] == None:
                preds[idx] = random.choice(['A', 'B', 'C', 'D'])
                cnt += 1
                print(idx + 1, 'failed to extract answer')
                print(repr(data[idx]['response']))
                print('====')
        correct_predictions = sum([1 for pred, gt in zip(preds, ground_truth) if pred == gt])
        total_predictions = len(ground_truth)
        accuracy = correct_predictions / total_predictions
        return {
            'acc': accuracy * 100, 'fail': 100 * cnt / len(preds)
        }

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--src-file', type=str, default="output/repulsar_sft-voicebench-openbookqa-test.jsonl")
    args = parser.parse_args()
    data = []
    with open(args.src_file, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())  # Convert JSON string to dictionary
            data.append(json_obj)
    results = MCQEvaluator().evaluate(data)
    print(results)