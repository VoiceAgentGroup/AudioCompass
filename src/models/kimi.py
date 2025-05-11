from .base import VoiceAssistant
import soundfile as sf
import os
import tempfile
from huggingface_hub import snapshot_download
from .src_kimi.api.kimia import KimiAudio
from torch.nn import CrossEntropyLoss

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

class KimiAssistant(VoiceAssistant):
    def __init__(self, **kwargs):
        self.model_name = 'kimi'
        cache_dir = os.path.join(kwargs.get('cache_dir', './cache'), 'models')
        self.model_path = os.path.join(cache_dir, "Kimi-Audio-7B-Instruct")
        if not os.path.exists(self.model_path):
            snapshot_download(
                repo_id="moonshotai/Kimi-Audio-7B-Instruct",
                local_dir=cache_dir,
            )
        self.model = KimiAudio(model_path=self.model_path, load_detokenizer=True, cache_dir=cache_dir)
        
    def asr(self, audio):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')
        messages_asr = [
            # You can provide context or instructions as text
            {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
            # Provide the audio file path
            {"role": "user", "message_type": "audio", "content": temp_filename}
        ]
        _, text_output = self.model.generate(messages_asr, **sampling_params, output_type="text")
        return text_output
        
    def generate_a2a(self, audio, max_new_tokens=2048):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')
        messages_conversation = [
            # Start conversation with an audio query
            {"role": "user", "message_type": "audio", "content": temp_filename}
        ]
        wav_output, _ = self.model.generate(messages_conversation, **sampling_params, output_type="both")
        return wav_output.detach().cpu().view(-1).numpy(), 24000
        
    def generate_a2t(self, audio, max_new_tokens=2048):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')
        messages_conversation = [
            # Start conversation with an audio query
            {"role": "user", "message_type": "audio", "content": temp_filename}
        ]
        _, text_output = self.model.generate(messages_conversation, **sampling_params, output_type="both")
        return text_output
    
    def generate_at2t(self, audio, text, max_new_tokens=2048):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')
        messages_conversation = [
            {"role": "user", "message_type": "audio", "content": temp_filename},
            {"role": "user", "message_type": "text", "content": text}
        ]
        _, text_output = self.model.generate(messages_conversation, **sampling_params, output_type="both")
        return text_output
    
    def get_ppl(self, input, input_type: str):
        output_type = 'both'
        if input_type == 'text':
            messages_conversation = [
                {"role": "user", "message_type": "text", "content": input}
            ]
            _, labels, _ = self.model.prompt_manager.get_prompt(messages_conversation, output_type=output_type).to_tensor()
        elif input_type == 'audio':
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
                temp_filename = temp_file.name
                # Write the audio data to the file
                sf.write(temp_file.name, input['array'], input['sampling_rate'], format='wav')
            messages_conversation = [
                # Start conversation with an audio query
                {"role": "user", "message_type": "audio", "content": temp_filename}
            ]
            labels, _, _ = self.model.prompt_manager.get_prompt(messages_conversation, output_type=output_type).to_tensor()
        else:
            raise ValueError("Invalid input_type", input_type)
        
        audio_input_ids, text_input_ids,  _ = self.model.prompt_manager.get_prompt(messages_conversation, output_type=output_type).to_tensor()
        labels = text_input_ids if input_type == 'text' else audio_input_ids
        lm_logits = self.model.alm.forward(audio_input_ids, text_input_ids).logits[0].contiguous().float()
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss.item()