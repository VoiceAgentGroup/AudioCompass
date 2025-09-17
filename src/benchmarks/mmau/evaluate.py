import argparse
import json
from tqdm import tqdm
from src.utils.matcher import string_match

def _evaluate(data):
    corr, total = 0, 0
    result_string = ""

    # Track metrics for different categories:
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    
    # Here is the new dict for sub-category metrics
    subcat_metrics = {}

    output_key = 'model_prediction' # The key that contains model output
    no_pred_count = 0

    for sample in tqdm(data):
        
        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        task = sample['task']
        difficulty = sample['difficulty']
        choices = sample['choices']
        
        # Get the sub-category
        subcat = sample.get('sub-category', None)
        if subcat is not None:
            # If we haven't seen this sub-category before, initialize
            if subcat not in subcat_metrics:
                subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            corr += 1

        total += 1
        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1

    # Format results into a string instead of printing:
    result_string += "*"*30 + "\n"
    result_string += "Task-wise Accuracy:\n"
    for task in task_metrics:
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        result_string += f"{task} : {acc:.2f}% over {n_total} samples\n"
    
    result_string += "*"*30 + "\n"
    result_string += "Difficulty-wise Accuracy:\n"
    for diff in diff_metrics:
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        result_string += f"{diff} : {acc:.2f}% over {n_total} samples\n"
    
    result_string += "*"*30 + "\n"
    result_string += "Sub-category-wise Accuracy:\n"
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        result_string += f"{subcat} : {acc:.2f}% over {n_total} samples\n"

    result_string += "*"*30 + "\n"
    result_string += f"Total Accuracy: {(corr/total) * 100:.2f}% over {total} samples\n"
    result_string += "*"*30 + "\n"
    result_string += f"No prediction count: {no_pred_count}\n"
    
    return result_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process benchmark JSON and calculate accuracy.")
    parser.add_argument('--src-file', type=str, required=True, help='Path to input JSON file to be evaluated')
    args = parser.parse_args()  
    
    with open(args.src_file, 'r') as f:
        data = json.load(f)

    results = _evaluate(data)
    print(results)
