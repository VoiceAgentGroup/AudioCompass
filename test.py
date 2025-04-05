import torch
import torchaudio
import json
from argparse import ArgumentParser
from src.models import load_model

choice_strs = ['a', 'b', 'c', 'd']

def main(args):
    for choice_str in choice_strs:
        question_path = f"datas/cmmlu-minimax/test/{args.s}/{args.i}/question_0.wav"
        choice_path = f"datas/cmmlu-minimax/test/{args.s}/{args.i}/choice_{choice_str}_0.wav"

        question, sample_rate = torchaudio.load(question_path)
        question = question.squeeze(0)
        choice, sample_rate = torchaudio.load(choice_path)
        choice = choice.squeeze(0)

        complete_audio = torch.cat([question, choice])

        complete_audio = {
            'array': complete_audio.numpy(),
            'sampling_rate': sample_rate,
        }

        model = load_model('local')
        _, response = model.generate_audio(complete_audio)

        with open('testrst_' + choice_str + '.json', 'w') as f:
            json.dump(response, f, indent=4)

        logprob = 0
        for pair in response[1:]:
            for token in pair.keys():
                logprob += pair[token]['logprob']
        logprob /= len(response) - 1

        print(f"choice_{choice_str} logprob: ", logprob)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', type=str, default='anatomy')
    parser.add_argument('-i', type=int)
    args = parser.parse_args()
    main(args)