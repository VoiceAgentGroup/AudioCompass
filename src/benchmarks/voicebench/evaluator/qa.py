import numpy as np
import multiprocessing
from qa_metrics.pedant import PEDANT
from tqdm import tqdm
from .base import Evaluator
from api_judge import generate


def majority_vote(scores):
    scores = [item.lower() for item in scores]
    final_answer = max(set(scores), key=scores.count)

    # Convert the final answer to True for 'Yes' and False for 'No'
    return True if final_answer == 'yes' else False


class QAEvaluator(Evaluator):
    def __init__(self):
        self.pedant = PEDANT()

    def evaluate(self, data):
        with multiprocessing.Pool(4) as pool:
            judged_data = list(tqdm(pool.imap(generate, data), total=len(data)))
        panda_results = [self.pedant.evaluate([item['reference'].lower()], item['response'].lower(), item['prompt'].lower()) for item in judged_data]
        gpt_results = [majority_vote(item['score']) for item in judged_data]
        return {
            'panda': np.mean(panda_results)*100, 'gpt': np.mean(gpt_results) * 100
        }
