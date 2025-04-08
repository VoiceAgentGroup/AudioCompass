import torch
import torchaudio
import os
import json
from argparse import ArgumentParser
from src.models import load_model

model = load_model('localhost')

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
    
    choice_text = []
    
    for choice in meta_data['QA']['choice'].keys():
        choice_str = choice + '.' + meta_data['QA']['choice'][choice][0]['text']
        choice_text.append(choice_str)

        choice_path = os.path.join(audio_dir, meta_data['QA']['choice'][choice][0]['file'])
        choice, sample_rate = torchaudio.load(choice_path)
        choice = choice.squeeze(0)

        complete_audio = torch.cat([question, choice])

        complete_audio = {
            'array': complete_audio.numpy(),
            'sampling_rate': sample_rate,
        }

        _, logprob_response = model.generate_audio(complete_audio)

        logprob = 0
        length = 0
        for token in logprob_response[1:]:
            token = list(token.values())[0]
            if token['decoded_token'] == '<|AUDIO|>':
                logprob += token['logprob']
                length += 1
                
        logprob /= length

        print(f"choice_{choice_str} logprob:", logprob, "audio_token_len:", length)
        
    choice_text = ' '.join(choice_text)
    prompt = 'Here is an audio of a question. Please listen carefully and choose the right answer.\n' + choice_text
    question = {
        'array': question.numpy(),
        'sampling_rate': sample_rate,    
    }
    response = model.generate_mixed(question, prompt)
    print('Normal response:\n' + response)
    
    print('Right Answer: ' + right_answer)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--subject', type=str, default='anatomy')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    main(args)