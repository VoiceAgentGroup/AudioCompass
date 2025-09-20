import traceback
import re
from tqdm import tqdm
from src.utils.client import AIClient
from src.utils.matcher import string_match

class Evaluator:
    def __init__(self, split):
        self.split = split

    def build_eval_messages(self, data):
        instruction = data["question"]
        targets = data["answers"]
        answer_to_be_judged = data["response"]

        if self.split == 'trivia_qa' or 'web_questions':
            prompt = f"""
Your will be given a question, the reference answers to that question, and an answer to be judged. Your tasks is to judge whether the answer to be judged is correct, given the question and reference answers. An answer considered correct expresses or contains the same meaning as at least **one of** the reference answers. The format and the tone of the response does not matter.  

You should respond in JSON format. First provide a one-sentence concise analysis for the judgement in field ‘analysis‘, then your judgment in field ‘judgment‘. For example, 
'''json 
{{"analysis": "<a one-sentence concise analysis for the judgement>", "judgment": < your final judgment, "correct" or "incorrect">}} 
'''  

# Question 
{instruction}  

# Reference Answer 
{targets}  

# Answer To Be Judged 
{answer_to_be_judged}

"""
        else:
            prompt = f"""
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
{targets}

## Model's Answer:
{answer_to_be_judged}

"""
        return prompt
    
    def get_eval_score(self, item):
        try:
            eval_js = eval(item[7:-3])
        except:
            eval_js = eval(item)
        assert "analysis" in eval_js and "judgment" in eval_js and eval_js["judgment"] in ["correct", "incorrect"]
        return eval_js
    
    def check_eval_response_format(self, item):
        try:
            item = self.get_eval_score(item)
            return item
        except Exception as e:
            traceback.print_exc()
            return {"judgment": ""}
    
    def llm_evaluate(self, datas):
        judge = AIClient('gpt-5')
        correct_count = 0
        for data in tqdm(datas):
            message = self.build_eval_messages(data)
            judged_result = judge.generate(message)
            if self.split == 'trivia_qa' or 'web_questions':
                judged_result = self.check_eval_response_format(judged_result)
                if judged_result["judgment"] == "correct":
                    correct_count += 1
            else:
                try:
                    score = re.findall(r"[Tt]he score is \[(Correct|Incorrect)\]", judged_result)[0]
                    if score == "correct":
                        correct_count += 1
                except:
                    print(f"Error parsing score from response: {judged_result}")
                    print(traceback.format_exc())
                    # assert 0
            print(f'correct_count: {correct_count}\n')
        return {'acc': correct_count / len(datas)}

    def rule_evaluate(self, datas):
        correct_count = 0
        for data in datas:
            correct = True
            for answer in data['answers']:
                match_result = string_match(answer=answer, prediction=data['response'])
                if not match_result:
                    correct = False
                    break
            if correct:
                correct_count += 1
        return {'acc': correct_count / len(datas)}