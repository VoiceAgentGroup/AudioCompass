import torch
import torchaudio
import os
import json
from argparse import ArgumentParser
from src.models import load_model

model = load_model('speechgpt2')
model.model.process_greeting()

def main(args):
    subject_dir = f'datas/cmmlu-minimax/test/{args.subject}'
    meta_path = os.path.join(subject_dir, f'{args.subject}_structured.json')
    audio_dir = os.path.join(subject_dir, str(args.idx))
    
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    meta_data = meta_data[args.idx]
    
    question_path = os.path.join(audio_dir, meta_data['QA']['question'][0]['file'])
    question, sample_rate = torchaudio.load(question_path)
    question = question.squeeze(0)
    
    question_text = meta_data['QA']['question'][0]['text']
    print('Question: ' + question_text)
    
    right_answer = meta_data['QA']['right_answer']
    
    for choice in meta_data['QA']['choice'].keys():

        choice_path = os.path.join(audio_dir, meta_data['QA']['choice'][choice][0]['file'])
        choice, sample_rate = torchaudio.load(choice_path)
        choice = choice.squeeze(0)

        complete_audio = torch.cat([question, choice])

        complete_audio = {
            'array': complete_audio.numpy(),
            'sampling_rate': sample_rate,
        }

        ppl = model.get_ppl(complete_audio, input_type='audio')
        # response = model.generate_a2t(complete_audio)

        print(f"{choice}-ppl:", ppl)
        # print(response)
    
    print('Right Answer: ' + right_answer)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--subject', type=str, default='anatomy')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    main(args)