import os
import json

split = 'test'

def gen_meta(base_path):
    cwd = os.getcwd()
    os.chdir(base_path)
    
    dir_list = os.listdir(split)
    dir_list.sort(key=lambda x: x[0])

    aggregated_datas = []
    for dir_path in dir_list:
        aggregated_data = {'subject': dir_path}
        json_path = os.path.join(split, dir_path, f'{dir_path}_structured.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Data file {json_path} not found.")
        with open(json_path, 'r') as f:
            data = json.load(f)
            aggregated_data['qa'] = []
            for item in data:
                qa = {}
                qa['question'] = {
                    'path': os.path.join(split, dir_path, item['idx'], item['QA']['question'][0]['file']),
                    'text': item['QA']['question'][0]['text']
                }
                choice_strs = ['A', 'B', 'C', 'D']
                qa['choice'] = [{
                    'path': os.path.join(split, dir_path, item['idx'], item['QA']['choice'][i][0]['file']),
                    'text': item['QA']['choice'][i][0]['text']
                    } for i in choice_strs]
                qa['right_answer'] = item['QA']['right_answer']
                aggregated_data['qa'].append(qa)
        aggregated_datas.append(aggregated_data)

    with open('meta_data.json', 'w') as f:
        json.dump(aggregated_datas, f, indent=4, ensure_ascii=False)
        
    os.chdir(cwd)
    
    return aggregated_datas
    
if __name__ == '__main__':
    gen_meta(base_path='datas/cmmlu-minimax')