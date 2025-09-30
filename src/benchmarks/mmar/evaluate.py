from src.utils.judge import string_match

def _evaluate(input_data):
    corr, total = 0, 0
    result_string = ""

    # Track metrics for different categories:
    modality_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0], 'mix-sound-music': [0, 0], 'mix-sound-speech': [0, 0], 'mix-music-speech': [0, 0], 'mix-sound-music-speech': [0, 0]}
    category_metrics = {'Signal Layer': [0, 0], 'Perception Layer': [0, 0], 'Semantic Layer': [0, 0], 'Cultural Layer': [0, 0]}
    
    # Here is the new dict for sub-category metrics
    subcat_metrics = {}

    output_key = 'model_prediction' # The key that contains model output
    no_pred_count = 0
    matched_outputs = []
    new_data = []

    # for idx, sample in enumerate(tqdm(input_data)):
    for idx, sample in enumerate(input_data):
        
        # If there's no model output key, skip
        if output_key not in sample:
            continue
        
        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        modality = sample['modality']
        category = sample['category']
        choices = sample['choices']
        
        # Get the sub-category
        subcat = sample.get('sub-category', None)
        if subcat is not None:
            # If we haven't seen this sub-category before, initialize
            if subcat not in subcat_metrics:
                subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            modality_metrics[modality][0] += 1
            category_metrics[category][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        total += 1
        new_data.append(sample)
        modality_metrics[modality][1] += 1
        category_metrics[category][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1


    # Print results:
    result_string == "*"*30 + "\n"
    print("*"*30)
    result_string += "Modality-wise Accuracy:\n"
    for modality in modality_metrics:
        n_correct, n_total = modality_metrics[modality]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        result_string += f"{modality} : {acc:.2f}% over {n_total} samples"
    
    result_string == "*"*30 + "\n"
    result_string += "Category-wise Accuracy:\n"
    for category in category_metrics:
        n_correct, n_total = category_metrics[category]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        result_string += f"{category} : {acc:.2f}% over {n_total} samples"
    
    result_string == "*"*30 + "\n"
    result_string += "Sub-category-wise Accuracy:\n"
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        result_string += f"{subcat} : {acc:.2f}% over {n_total} samples"

    result_string == "*"*30 + "\n"
    result_string += f"Total Accuracy: {(corr/total) * 100:.2f}% over {total} samples"
    result_string == "*"*30 + "\n"
    result_string += f"No prediction count: {no_pred_count}"
    
    return result_string