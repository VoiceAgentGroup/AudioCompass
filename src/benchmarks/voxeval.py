import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from loguru import logger
import json
from .base import BaseBenchmark
from src.transcriptors import WhisperLargeV3
from src.utils.rule_extractor import extract_answer


class VoxEval(BaseBenchmark):
    def __init__(self, timbre='alloy', data_dir="datas/VoxEval", cache_dir='cache', **kwargs):
        self.name = 'voxeval'
        self.data_dir = os.path.join(cache_dir, data_dir)
        
        # Parameters
        self.check_timbre(timbre)
        self.shots = kwargs.get("shots", 5)
        self.prompt_mode = kwargs.get("prompt_mode", "regular")
        self.cut_audio = kwargs.get("cut_audio", True)
        
        self.test_dir = os.path.join(self.data_dir, "test")
        self.fewshot_dir = os.path.join(self.data_dir, "all_fewshot_examples")
        
        # Auto-configure based on prompt_mode
        if self.prompt_mode == "CoT":
            self.shots = kwargs.get("shots", 3)
            self.fewshot_dir = os.path.join(self.data_dir, "math_CoT_fewshot")
        
        self.transcriptor = WhisperLargeV3(**kwargs)
        
        self.dataset = self.load_data(**kwargs)
        logger.add(f'log/{self.name}-{self.timbre}.log', rotation='50MB')
        
    def check_timbre(self, timbre):
        available_timbre = ['alloy', 'echo', 'fable', 'nova', 'onyx', 'shimmer']
        if timbre not in available_timbre:
            raise ValueError("Timbre should be one of " + available_timbre)
        self.timbre = timbre
    
    def concat_audio(self, audio_paths, add_silence=True):
        audio_segments = []
        sample_rate = None
        
        # Load all audio segments
        for path in audio_paths:
            audio, sr = torchaudio.load(path)
            if sample_rate is None:
                sample_rate = sr
            audio_segments.append(audio)
        
        # Add silence between segments if requested
        if add_silence and len(audio_segments) > 1:
            silence = torch.zeros((audio_segments[0].shape[0], sample_rate))  # 1 second silence
            combined = audio_segments[0]
            
            for segment in audio_segments[1:]:
                combined = torch.cat((combined, silence, segment), dim=1)
        else:
            combined = torch.cat(audio_segments, dim=1)
            
        return combined
        
    def load_data(self, **kwargs):
        logger.info("Preparing VoxEval data...")
        
        # Determine subject list based on prompt mode
        if self.prompt_mode == "CoT":
            subject_list = ["elementary_mathematics_4o.csv", "high_school_mathematics_4o.csv", "college_mathematics_4o.csv"]
        else:
            subject_list = []
            for item in os.listdir(self.test_dir):
                if item.endswith(".csv"):
                    subject_list.append(item)
        
        logger.info(f"Found {len(subject_list)} subjects: {subject_list}")
        
        prepared_data = []
        few_shot_prompts = {}
        
        # First prepare the few-shot prompts for each subject type
        for subject in subject_list:
            folder = subject.split(".csv")[0]
            if self.prompt_mode == "CoT":
                fewshot_folder = folder.replace("_4o", "_dev_4o")
            else:
                fewshot_folder = folder.replace("_test", "_val")
            
            if fewshot_folder not in few_shot_prompts:
                logger.info(f"Preparing few-shot prompt for {fewshot_folder}")
                try:
                    # Collect paths for few-shot examples
                    few_shot_paths = []
                    for i in range(self.shots):
                        formatted_i = "%08d" % i
                        question_i = f"{formatted_i}_question.mp3"
                        if self.prompt_mode == "CoT":
                            answer_i = f"{formatted_i}_CoT_answer.mp3"
                        else:
                            answer_i = f"{formatted_i}_answer.mp3"
                        
                        question_path = os.path.join(self.fewshot_dir, self.timbre, fewshot_folder, question_i)
                        answer_path = os.path.join(self.fewshot_dir, self.timbre, fewshot_folder, answer_i)
                        
                        few_shot_paths.append(question_path)
                        few_shot_paths.append(answer_path)
                    
                    # Concatenate all few-shot examples
                    few_shot_audio = self.concat_audio(few_shot_paths)
                    few_shot_prompts[fewshot_folder] = few_shot_audio
                    
                except Exception as e:
                    logger.error(f"Error creating few-shot prompt for {fewshot_folder}: {e}")
                    few_shot_prompts[fewshot_folder] = None
        
        # Now prepare each subject's questions
        for subject in subject_list:
            logger.info(f"Preparing questions for {subject}")
            folder = subject.split(".csv")[0]
            if self.prompt_mode == "CoT":
                fewshot_folder = folder.replace("_4o", "_dev_4o")
            else:
                fewshot_folder = folder.replace("_test", "_val")
            
            # Get the few-shot prompt for this subject
            few_shot_prompt = few_shot_prompts.get(fewshot_folder)
            if few_shot_prompt is None:
                logger.error(f"No few-shot prompt available for {subject}, skipping.")
                continue
                
            # Open the subject csv file
            try:
                csv_path = os.path.join(self.test_dir, subject)
                df = pd.read_csv(csv_path, header=None)
            except Exception as e:
                logger.error(f"Error loading CSV for {subject}: {e}")
                continue
                
            # Prepare each question
            for i in range(len(df)):
                formatted_i = "%08d" % i
                question_i = f"{formatted_i}_question.mp3"
                question_path = os.path.join(self.test_dir, self.timbre, folder, question_i)
                
                if not os.path.exists(question_path):
                    logger.warning(f"Question audio not found for {subject}, question {i}")
                    continue
                
                try:
                    question_audio, sampling_rate = torchaudio.load(question_path)
                    
                    # Create silence (1 second)
                    silence = torch.zeros((question_audio.shape[0], 16000))  # Standard 16kHz sample rate
                    
                    # Concat with few-shot prompt
                    input_audio = torch.cat((few_shot_prompt, silence, question_audio), dim=1)
                    
                    # Cut audio if needed
                    if self.cut_audio:
                        input_audio = input_audio[:, -80 * 16000:]  # Last 80 seconds at 16kHz
                    
                    input_audio = {
                        'array': input_audio.squeeze(0).numpy(),
                        'sampling_rate': sampling_rate,
                    }
                    
                    # Add to prepared data
                    prepared_data.append({
                        'subject': subject,
                        'question_id': i,
                        'question': df.iloc[i, 1],
                        'options': df.iloc[i, 2],
                        'answer': df.iloc[i, 3],
                        'audio': input_audio
                    })
                    
                except Exception as e:
                    logger.error(f"Error preparing question {i} in {subject}: {e}")
                    continue
        
        logger.info(f"Prepared {len(prepared_data)} questions across {len(subject_list)} subjects")
        return prepared_data
    
    def generate(self, model):
        logger.info("Generating VoxEval results...")
        
        results = []
        failed_items = []
        
        for idx, item in enumerate(tqdm(self.dataset)):
            logger.info(f"Processing timbre {self.timbre} subject {item['subject']}, question {item['question_id']}")
            
            try:
                input_audio = item['audio']
                response_audio, sample_rate = model.generate_a2a(input_audio)
                transcription = self.transcriptor.inference(response_audio, generate_kwargs={"language": "english"})
                
                result_item = {k: v for k, v in item.items() if k != 'audio'}
                result_item['idx'] = idx
                result_item['response_audio'] = response_audio
                result_item['sample_rate'] = sample_rate
                result_item['response'] = transcription
                results.append(result_item)
                
                logger.info(f"Response: {transcription}")
                logger.info('====================================')
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                failed_items.append([item['subject'], item['question_id'], f"generation error: {e}"])
                continue
        
        if failed_items:
            logger.warning(f"Failed items: {len(failed_items)}")
            for item in failed_items:
                logger.error(f"Failed: {item}")
        
        return results
    
    def evaluate(self, data):
        logger.info("Evaluating VoxEval results...")
        
        results = {"overall": {}, "by_subject": {}}
        
        if not data:
            logger.warning("No data to evaluate")
            return results
        
        # Group results by subject
        subject_results = {}
        for item in data:
            subject = item['subject']
            if subject not in subject_results:
                subject_results[subject] = []
            subject_results[subject].append(item)
        
        # Calculate accuracy for each subject
        total_correct = 0
        total_questions = 0
        
        for subject, items in subject_results.items():
            correct = 0
            for item in items:
                # Extract the answer from the response
                response = item['response'].lower()
                answer = item['answer'].lower()
                answer = extract_answer(answer)
                
                # Simple exact match for now (can be improved later)
                if answer == response:
                    correct += 1
            
            accuracy = correct / len(items) if items else 0
            results["by_subject"][subject] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(items)
            }
            
            total_correct += correct
            total_questions += len(items)
        
        # Calculate overall accuracy
        overall_accuracy = total_correct / total_questions if total_questions else 0
        results["overall"] = {
            "accuracy": overall_accuracy,
            "correct": total_correct,
            "total": total_questions
        }
        
        return results
    
    def save_generated_results(self, results, output_dir, model_name):
        logger.info(f"Saving VoxEval results...")
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, self.name)
        wav_dir = os.path.join(output_dir, 'wavs')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        
        for result in results:
            wav_path = os.path.join(wav_dir, f"{model_name}-{self.timbre}-{result['idx']}.wav")
            wav = result['response_audio'] if result['response_audio'].ndim == 2 else result['response_audio'].unsqueeze(0)
            torchaudio.save(wav_path, wav, result['sample_rate'])
            result['response_audio_path'] = wav_path
            result.pop('response_audio')
            result.pop('sample_rate')
            
        model_name = model_name.split('/')[-1]
        results_file = os.path.join(output_dir, f"{model_name}-{self.name}-{self.timbre}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {output_dir}")
    
    def run(self, model, output_dir):
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        evaluation_results = self.evaluate(generated_results)
        return evaluation_results