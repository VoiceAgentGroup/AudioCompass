import re
import traceback
import multiprocessing
import numpy as np
from tqdm import tqdm
from src.utils.client import AIClient
from .config import judge_model

class AlpacaEvaluator:

    def __init__(self):
        self.final_output_path = "output/alpaca_eval_results.xlsx"

    def build_eval_messages(self, data):
        instruction = data["instruction"]
        response = data["infer_response"]
        pattern = f"""
        [Instruction] 
        Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".  
        [Question] 
        {instruction}  
        
        [The Start of Assistant’s Answer] 
        {response} 
        [The End of Assistant’s Answer]
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": pattern}]}]
        return messages
    
    def get_eval_score(self, item):
        try:
            # 使用正则表达式匹配 [[数字]] 格式
            match = re.search(r'\[\[(\d+)\]\]', item)
            if match:
                score = int(match.group(1))
                # 确保分数在1-10范围内
                assert 1 <= score <= 10
                return score
        except:
            # 如果解析失败，打印错误信息并返回错误状态
            print(f"Error parsing score from response: {item}")
            print(traceback.format_exc())
            assert 0
    
    def evaluate(self, datas):
        messages = []
        for data in datas:
            messages.append(self.build_eval_messages(data))
        
        judge = AIClient()
        with multiprocessing.Pool(4) as pool:
            judged_data = list(tqdm(pool.imap(judge.generate, judge_model, messages), total=len(messages)))
        
        scores = []
        for item in judged_data:
            score = self.get_eval_score(item)
            if score:
                scores.append(score)
        return {'gpt': np.mean(scores)}
