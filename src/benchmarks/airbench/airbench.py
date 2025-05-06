import os
import json
from tqdm import tqdm
from loguru import logger
import pandas as pd
import torchaudio
from ..base import BaseBenchmark
from src.utils.ai_judge import OPENAI_Judge

class AIRBench(BaseBenchmark):
    """
    Benchmark for AIR-Bench tasks supporting two inference modes:
      - chat: simple question-response style with AI judge scoring
      - foundation: multiple-choice style with exact-match accuracy
    """
    def __init__(self, split, data_dir="datas/AIR-Bench", cache_dir='cache', **kwargs):
        self.name = 'airbench'
        self.split = split  # 'chat' or 'foundation'
        self.data_dir = os.path.join(cache_dir, data_dir)

        # Logging
        logger.add(f'log/{self.name}-{self.split}.log', rotation='50MB')

        # Load dataset metadata
        self.dataset = self.load_data()

    def load_data(self):
        logger.info("Loading AIR-Bench data...")
        # Determine meta file path based on mode
        if self.split == 'chat':
            meta_file = os.path.join(self.data_dir, 'Chat', 'Chat_meta.json')
        else:
            meta_file = os.path.join(self.data_dir, 'Foundation', 'Foundation_meta.json')

        if not os.path.exists(meta_file):
            logger.error(f"Meta file not found: {meta_file}")
            return []
        with open(meta_file, 'r', encoding='utf-8') as f:
            records = json.load(f)

        prepared = []
        for item in records:
            task = item.get('task_name')
            dataset_name = item.get('dataset_name')
            wav = item.get('path')
            # Handle flac for grounding
            if self.split == 'foundation' and task == 'Audio_Grounding':
                wav = wav[:-3] + 'flac'
            audio_path = os.path.join(self.data_dir, self.split.capitalize(), f"{task}_{dataset_name}", wav)
            if not os.path.exists(audio_path):
                logger.warning(f"Missing audio file: {audio_path}")
                continue
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as e:
                logger.error(f"Error loading audio file {audio_path}: {e}")
                continue
            audio = {
                'array': waveform.squeeze(0).numpy(),
                'sampling_rate': sample_rate
            }
            item['audio'] = audio
            prepared.append(item)

        logger.info(f"Loaded {len(prepared)} examples for mode '{self.split}'")
        return prepared

    def generate(self, model):
        logger.info("Starting generation for AIR-Bench...")
        # Store model name for evaluation
        self.model_name = getattr(model, 'model_name', None)

        results = []
        for rec in tqdm(self.dataset, desc="AIR-Bench Inference"):
            audio = rec['audio']
            question = rec.get('question', '')
            logger.info(f"Processing record: {rec['task_name']}, question: {question}")
            try:
                if self.split == 'chat':
                    instruction = question
                    output = model.generate_at2t(audio, instruction)
                else:
                    # single-choice foundation
                    prompt = (
                        'Choose the most suitable answer from A, B, C, D. ' \
                        'Provide only the option letter.'
                    )
                    choices = [f"A. {rec.get('choice_a')}", f"B. {rec.get('choice_b')}"]
                    if rec.get('choice_c') is not None:
                        choices.append(f"C. {rec.get('choice_c')}")
                    if rec.get('choice_d') is not None:
                        choices.append(f"D. {rec.get('choice_d')}")
                    instruction = '\n'.join([prompt, question] + choices)
                    output = model.generate_at2t(audio, instruction)
                rec['response'] = output.strip()
                logger.info(f"Generated response: {output}")
                logger.info('====================================')
                results.append(rec)
            except Exception as e:
                logger.error(e)
                logger.error('====================================')
                continue
        return results

    def evaluate(self, results):
        logger.info("Evaluating AIR-Bench results...")
        summary = {'overall': {}, 'by_task': {}}
        if not results:
            logger.warning("No results to evaluate.")
            return summary

        if self.split == 'chat':
            # Use AI judge for scoring chat-style
            judge = OPENAI_Judge()
            total_score = 0
            total_count = 0
            task_scores = {}
            for rec in results:
                task = rec['task_name']
                # get audio meta info
                if rec.get('meta_info', None) == None:
                    print("lack meta info")
                    exit(1)
                else:
                    meta_info = rec['meta_info']
                prompt = (
                    "You are a helpful and precise assistant for checking the quality of the answer.\n"
                        f"[Detailed Audio Description]\n{meta_info}\n[Question]\n{rec.get('question')}\n"
                        f"[The Start of Assistant 1s Answer]\n{rec.get('answer_gt')}\n[The End of Assistant 1s Answer]\n"
                        f"[The Start of Assistant 2s Answer]\n{rec.get('response')}\n[The End of Assistant 2s Answer]\n[System]\n"
                        "We would like to request your feedback on the performance of two AI assistants in response to the user question "
                        "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
                        "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
                        "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
                        "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
                        "The two scores are separated by a space." 
                )
                try:
                    score_str = judge.generate(self.model_name, prompt)
                    score = float(score_str)
                except Exception as e:
                    logger.error(f"Judge error for record {rec}: {e}")
                    score = 0.0
                total_score += score
                total_count += 1
                if task not in task_scores:
                    task_scores[task] = {'score': 0, 'count': 0}
                task_scores[task]['score'] += score
                task_scores[task]['count'] += 1

            overall = total_score / total_count if total_count else 0
            summary['overall'] = {'avg_score': overall, 'count': total_count}
            for task, stats in task_scores.items():
                avg = stats['score'] / stats['count'] if stats['count'] else 0
                summary['by_task'][task] = {'avg_score': avg, 'count': stats['count']}
        else:
            # Exact-match accuracy for foundation-style
            task_stats = {}
            total = len(results)
            correct = 0
            for rec in results:
                task = rec['task_name']
                gt = rec.get('answer_gt', '').strip().lower()
                pred = rec.get('response', '').strip().lower()
                is_correct = (gt == pred)
                correct += int(is_correct)
                if task not in task_stats:
                    task_stats[task] = {'correct': 0, 'total': 0}
                task_stats[task]['correct'] += int(is_correct)
                task_stats[task]['total'] += 1
            overall_acc = correct / total if total else 0
            summary['overall'] = {'accuracy': overall_acc, 'correct': correct, 'total': total}
            for task, stats in task_stats.items():
                acc = stats['correct'] / stats['total'] if stats['total'] else 0
                summary['by_task'][task] = {'accuracy': acc, 'correct': stats['correct'], 'total': stats['total']}
        return summary

    def save_results(self, results, save_dir, model_name):
        logger.info("Saving AIR-Bench results...")
        out_file = os.path.join(save_dir, f"{model_name}-{self.split}.jsonl")
        with open(out_file, 'w', encoding='utf-8') as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def run(self, model, output_dir):
        save_dir = os.path.join(output_dir, self.name)
        os.makedirs(save_dir, exist_ok=True)
        
        results = self.generate(model)
        self.save_results(results, save_dir, model.model_name)

        summary = self.evaluate(results)
        summary_file = os.path.join(save_dir, f"{model.model_name}-{self.split}-eval.json")
        with open(summary_file, 'w', encoding='utf-8') as fsum:
            json.dump(summary, fsum, indent=2, ensure_ascii=False)
        return None