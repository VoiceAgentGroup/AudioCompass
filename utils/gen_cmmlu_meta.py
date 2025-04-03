import os
import json

base_path = '../datas/cmmlu-minimax'
os.chdir(base_path)

split_path = 'dev'
aggregated_datas = []

dir_list = os.listdir(split_path)
dir_list.sort(key=lambda x: x[0])

for dir_path in dir_list:
    aggregated_data = {'subject': dir_path}
    json_path = os.path.join(split_path, dir_path, f'{dir_path}_structured.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Data file {json_path} not found.")
    with open(json_path, 'r') as f:
        data = json.load(f)
        aggregated_data['qa'] = []
        for item in data:
            qa = {}
            qa['splits'] = item['splits']
            qa['pos'] = item['pos']
            qa['question'] = [os.path.join(split_path, dir_path, item['idx'], q['file']) for q in item['QA']['question']]
            choice_strs = ['A', 'B', 'C', 'D']
            qa['choice'] = [os.path.join(split_path, dir_path, item['idx'], item['QA']['choice'][i][0]['file']) for i in choice_strs]
            qa['right_answer'] = item['QA']['right_answer']
            aggregated_data['qa'].append(qa)
    aggregated_datas.append(aggregated_data)

with open('meta_data.json', 'w') as f:
    json.dump(aggregated_datas, f, indent=4)
print('Meta data generated successfully.')