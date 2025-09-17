import traceback
import multiprocessing
import numpy as np
from tqdm import tqdm
from src.utils.client import AIClient
from .config import judge_model

class WebQuestionsEvaluator:

    def __init__(self):
        self.final_output_path = "output/web_questions_results.xlsx"

    def build_eval_messages(self, data):
        instruction = data["question"]
        targets = data["answers"]
        answer_to_be_judged = data["infer_response"]

        pattern = f"""
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
        messages = [{"role": "user", "content": [{"type": "text", "text": pattern}]}]
        return messages
    
    def get_eval_score(self, item):
        try:
            eval_js = eval(item["eval_response"][7:-3])
        except:
            eval_js = eval(item["eval_response"])
        assert "analysis" in eval_js and "judgment" in eval_js and eval_js["judgment"] in ["correct", "incorrect"]
        return eval_js
    
    def check_eval_response_format(self, item):
        try:
            item = self.get_eval_score(item)
            return item
        except Exception as e:
            traceback.print_exc()
            return {"judgment": ""}
    
    def evaluate(self, datas):
        messages = []
        for data in datas:
            messages.append(self.build_eval_messages(data))
        
        judge = AIClient()
        with multiprocessing.Pool(4) as pool:
            judged_data = list(tqdm(pool.imap(judge.generate, judge_model, messages), total=len(messages)))
        judged_data = list(map(self.check_eval_response_format, judged_data))
        correct_count = 0
        for item in judged_data:
            if item["judgment"] == "correct":
                correct_count += 1
        return {'acc': correct_count / len(judged_data)}