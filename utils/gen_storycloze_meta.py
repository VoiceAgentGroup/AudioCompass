import os
import json

base_path = '../datas/storycloze'
os.chdir(base_path)

sSC_files = os.listdir('sSC')
sorted_sSC_files = sorted(sSC_files, key=lambda x: int(x.split('_')[0]))

tSC_files = os.listdir('tSC')
sorted_tSC_files = sorted(tSC_files, key=lambda x: int(x.split('_')[0]))

assert len(sorted_sSC_files) == len(sorted_tSC_files)

meta_datas = []

for i in range(0, len(sorted_sSC_files), 4):
    sSC_question = sorted_sSC_files[i: i + 4]
    tSC_question = sorted_tSC_files[i: i + 4]
    correct_txt, incorrect_sSC_txt = list(filter(lambda x: x.endswith('txt'), sSC_question))
    correct_wav, incorrect_sSC_wav = list(filter(lambda x: x.endswith('wav'), sSC_question))
    _, incorrect_tSC_txt = list(filter(lambda x: x.endswith('txt'), tSC_question))
    _, incorrect_tSC_wav = list(filter(lambda x: x.endswith('wav'), tSC_question))
    with open('sSC/' + correct_txt, 'r') as f:
        correct_text = f.read()
    with open('sSC/' + incorrect_sSC_txt, 'r') as f:
        incorrect_sSC_text = f.read()
    with open('tSC/' + incorrect_tSC_txt, 'r') as f:
        incorrect_tSC_text = f.read()
    meta_datas.append({
        'idx': i // 4,
        'correct': {
            'text': correct_text,
            'wav': 'sSC/' + correct_wav
        },
        'sSC': {
            'text': incorrect_sSC_text,
            'wav': 'sSC/' + incorrect_sSC_wav
        },
        'tSC': {
            'text': incorrect_tSC_text,
            'wav': 'tSC/' + incorrect_tSC_wav
        }
    })
    
with open('meta_data.json', 'w') as f:
    json.dump(meta_datas, f, indent=4)
print('Meta data generated successfully.')