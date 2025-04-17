import re
import traceback
import multiprocessing
import numpy as np
from tqdm import tqdm
from src.utils.ai_judge import OPENAI_Judge
from .config import judge_model

class LlamaQuestionsEvaluator:

    def __init__(self):
        self.final_output_path = "output/llama_questions_results.xlsx"
        
    def build_eval_messages(self, data):
        prompt = data["Questions"]
        gt_answer = data["Answer"]
        answer = data["infer_response"]

        eval_prompt = f"""
## Background
You are a professional QA evaluation expert. You need to assess whether the model's answer is correct based on the standard answer.\n\n

## Scoring Criteria
Correct: The answer matches or is equivalent to the standard answer \n
Incorrect: The answer is wrong or irrelevant to the question \n\n

## Evaluation Guidelines
1. The expression of answers can be flexible, not requiring exact matches. For example: \n
   - Numbers can be expressed in either Arabic numerals or words \n
   - Proper nouns can be in either English or Chinese \n
   - Differences in punctuation can be ignored \n
2. Focus on whether the core meaning of the answer is correct \n

## Output Format
Provide the reasoning for your score, then generate the result in "[]" format and make sure it contains "the score is [Correct]" or "the score is [Incorrect]", for example:
```
The answer is correct and equivalent to the standard answer, the score is [Correct]
```
or
```
The answer is incorrect and does not match the standard answer, the score is [Incorrect]
```
\n\n
## Question:
{prompt}

## Standard Answer:
{gt_answer}

## Model's Answer:
{answer}

"""
        messages = [{"role": "user", "content": eval_prompt}]
        return messages
    
    def evaluate(self, datas):
        messages = []
        for data in datas:
            messages.append(self.build_eval_messages(data))
        
        judge = OPENAI_Judge()
        with multiprocessing.Pool(4) as pool:
            judged_data = list(tqdm(pool.imap(judge.generate, judge_model, messages), total=len(messages)))
        correct_count = 0
        for item in judged_data:
            try:
                score = re.findall(r"[Tt]he score is \[(Correct|Incorrect)\]", item)[0]
                if score == "Correct":
                    correct_count += 1
            except:
                print(f"Error parsing score from response: {item}")
                print(traceback.format_exc())
                assert 0
        return {'acc': correct_count / len(datas)}