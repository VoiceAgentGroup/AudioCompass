from .evaluator import evaluator_mapping

def _evaluate(data, subset_name):
    evaluator = evaluator_mapping[subset_name]()
    results = evaluator.evaluate(data)
    if not results:
        raise ValueError("No results returned from evaluator.")
    return results