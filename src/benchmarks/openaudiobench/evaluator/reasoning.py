import re
import multiprocessing
import numpy as np
from tqdm import tqdm
from src.utils.ai_judge import OPENAI_Judge
from .config import judge_model

class ReasoningQAEvaluator:

    def __init__(self):
        self.final_output_path = "output/reasoning_qa_results.xlsx"

    def build_eval_messages(self, data):
        prompt = data["Prompt"]
        gt_answer = data["参考答案"]
        answer = data["infer_response"]

        eval_prompt = f"""
## 背景
现在你是一个大学数学老师。你需要依据 标准答案 来判断每道题的得分\n\n 

## 判分依据
5分答案：满分答案，需要回答的答案正确，同时过程正确，且回答考虑到了各种可能性，考虑全面 \n
4分答案：答案正确，但是没有过程 \n
3分答案：答案错误，过程大部分正确；或者答案正确，但是过程出现明显错误 \n
2分答案：答案错误，且过程大部分错误 \n
1分答案：答案错误，过程和思路全错\n\n 

## 其他注意事项
你需要忽略格式问题，以下都是一些等价的情况，不应该作为答案正确性的判断，比如 \n
1）latex格式表达的公式，普通格式表达的公式 \n
2）分数和小数表达的数值：比如1/3和0.33都算对 \n
3）关于π的表达：比如π、pi、3.14都是等价的 \n
4）关于常数的表达：比如n、k等常数表达都是等价的 \n
等，还有很多其他类似的等价表达 \n\n 

## 生成格式
以"[]"的格式生成分数，比如：
```
得分是[4]
```
\n\n 

## 题目
{prompt}

## 标准答案: 
{gt_answer}

## 学生回答:
{answer}

"""
        messages = [{"role": "user", "content": [{"type": "text", "text": eval_prompt}]}]
        return messages
    

    def evaluate(self, datas):
        messages = []
        for data in datas:
            messages.append(self.build_eval_messages(data))
        
        judge = OPENAI_Judge()
        with multiprocessing.Pool(4) as pool:
            judged_data = list(tqdm(pool.imap(judge.generate, judge_model, messages), total=len(messages)))
        scores = []
        for item in judged_data:
            res = re.findall(r'\[([0-5])\]', item)
            if len(res) >= 1:
                score = int(res[-1])
            else:
                score = -1
            scores.append(score)
        return {'gpt': np.mean(scores)}

            
        