import json
import re
import torch
import numpy as np
import torchaudio

from typing import Union
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
)

from .mimo_qwen2_grouped import *
from .Codec.models.codec import Generator as SpeechGPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MIMOStopper(StoppingCriteria):
    def __init__(
        self, stop_id: int, group_size: int, audio_channels: int, max_tokens: int
    ) -> None:
        super().__init__()
        self.stop_id = stop_id
        self.group_size = group_size
        self.audio_channels = audio_channels
        self.max_tokens = max_tokens

    def __call__(self, input_ids: torch.LongTensor, scores) -> bool:
        # Stop when last token of channel 0 is the stop token
        return (
            input_ids[0, -((self.audio_channels + 1) * self.group_size)].item()
            == self.stop_id
        ) or input_ids.numel() // self.group_size // (
            self.audio_channels + 1
        ) >= self.max_tokens


class InputSegment:
    def __init__(
        self,
        text: str = None,
        audio: torch.Tensor = None,
        tokenized_text: torch.Tensor = None,
        zeroemb_idx: int = 1024,  # TODO: Make this a parameter
        add_sosp_eosp=True,
        add_zeroemb_loss=False,
    ) -> None:
        has_text = text is not None
        has_tokenized_text = tokenized_text is not None
        assert has_text or has_tokenized_text, "Text channel cannot be empty"
        assert not (
            has_text and has_tokenized_text
        ), "Can't both have text and tokenized text"
        if has_tokenized_text:
            assert tokenized_text.shape[0] <= audio.reshape(-1, 3).shape[0]
        self.audio = audio
        self.text = text
        self.tokenized_text = tokenized_text
        self.zeroemb_idx = zeroemb_idx
        self.add_sosp_eosp = add_sosp_eosp

    @staticmethod
    def insert_between(tensor, i, value=-1):
        return torch.scatter(
            torch.full(
                (1, tensor.shape[1] + (tensor.shape[1] - 1) * i + i),
                value,
                dtype=tensor.dtype,
            ),
            1,
            torch.arange(0, tensor.shape[1], dtype=torch.int64)[None] * (i + 1),
            tensor,
        )

    def to_input_id(
        self,
        tokenizer,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenized_text is None:
            tokenized_text = tokenizer(
                self.text,
                return_tensors="pt",
                truncation=True,
                max_length=999999,
                padding=False,
                add_special_tokens=False,
            )[
                "input_ids"
            ].int()  # [1, seqlen]
        else:
            tokenized_text = self.tokenized_text.unsqueeze(0)

        if self.audio is None:  # Pure text block
            # Add group_size - 1 tokens between every two text tokens
            if group_size > 1:
                tokenized_text = self.insert_between(
                    tokenized_text, group_size - 1, value=-100
                )
            audio_part_input_id = torch.full(
                (3, tokenized_text.shape[1]), self.zeroemb_idx, dtype=torch.int
            )
        else:  # Audio + text block
            sosp_token = (
                tokenizer.convert_tokens_to_ids("<|sosp|>")
                if self.add_sosp_eosp
                else None
            )
            eosp_token = (
                tokenizer.convert_tokens_to_ids("<|eosp|>")
                if self.add_sosp_eosp
                else None
            )
            audio_part = self.audio.reshape(-1, 3).T  # [3, seqlen]
            assert (
                audio_part.shape[1] % group_size == 0
            ), f"Audio shape {audio_part.shape} is not divisible by group_size {group_size}"

            if tokenized_text.shape[1] * group_size > audio_part.shape[1]:
                print(
                    f"Expected text to be shorter than or equal to audio, but got text {tokenized_text.shape} * group_size and audio {audio_part.shape}"
                )
                tokenized_text = tokenized_text[:, : audio_part.shape[1] // group_size]
                print(f"Truncated text to {tokenized_text.shape} * group_size")
                print(f"The offending text is: {self.text}")

            if tokenized_text.shape[1] * group_size < audio_part.shape[1]:
                tokenized_text = F.pad(
                    tokenized_text,
                    (0, audio_part.shape[1] // group_size - tokenized_text.shape[1]),
                    value=tokenizer.convert_tokens_to_ids("<|empty|>"),
                ).int()
            tokenized_text = (
                torch.cat(
                    [
                        torch.tensor([[sosp_token]], dtype=torch.int),
                        tokenized_text,
                        torch.tensor([[eosp_token]], dtype=torch.int),
                    ],
                    dim=1,
                )
                if self.add_sosp_eosp
                else tokenized_text
            )
            tokenized_text = self.insert_between(
                tokenized_text, group_size - 1, value=-100
            )
            audio_part_input_id = (
                torch.cat(
                    [
                        torch.full((3, group_size), self.zeroemb_idx, dtype=torch.int),
                        audio_part,
                        torch.full((3, group_size), self.zeroemb_idx, dtype=torch.int),
                    ],
                    dim=1,
                )
                if self.add_sosp_eosp
                else audio_part
            )

        input_ids = torch.cat(
            [tokenized_text, audio_part_input_id], dim=0
        )  # [4, seqlen]

        return input_ids


class Inference:
    def __init__(
        self, model_path, model_args, codec_ckpt_path, codec_config_path
    ) -> None:
        self.device = DEVICE
        self.group_size = 3

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        padding_idx = self.tokenizer.pad_token_id
        self.sosp_idx = self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        self.eosp_idx = self.tokenizer.convert_tokens_to_ids("<|eosp|>")

        self.empty_token = self.tokenizer.convert_tokens_to_ids("<|empty|>")
        self.end_empty_token = self.tokenizer.convert_tokens_to_ids("<|end_empty|>")

        self.model = MIMOLlamaForCausalLM.from_pretrained(
            model_path,
            padding_idx=padding_idx,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            args=model_args,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()
        self.model = torch.compile(self.model, mode="reduce-overhead")

        self.generate_kwargs = {
            # "max_new_tokens": 5000,
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.9,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        }
        

        self.generator = SpeechGPT2Tokenizer.load_from_checkpoint(
            config_path=codec_config_path, checkpoint_path=codec_ckpt_path
        )
        self.generator = self.generator.to(self.device)
        self.generator.eval()
        self.generator = torch.compile(self.generator, mode="reduce-overhead")

        self.history = []

        self.greeting = None

    def set_greeting(self, text, audio):
        text = torch.tensor(text)
        audio = torch.tensor(audio).reshape(3, -1)
        self.greeting = [
            InputSegment(f"[|SpeechGPT|]: "),
            InputSegment(
                tokenized_text=text,
                audio=audio,
            ),
            InputSegment(f" ###\n{self.tokenizer.eos_token}"),
        ]

        greeting_audio_detokenized = self.generator.inference_detokenize(
            audio.reshape(-1, 3)
            .unsqueeze(0)
            .permute(2, 0, 1)
            .type(torch.LongTensor)
            .to(self.device)
        )
        return (
            24000,
            greeting_audio_detokenized.reshape(-1).detach().cpu().numpy(),
        )
                
    def process_greeting(self, greeting_source="src/models/src_speechgpt2/extra/greetings.jsonl", greeting_line_idx=2):
        greeting_line_idx = int(greeting_line_idx)
        with open(greeting_source, "r") as f:
            for idx, line in enumerate(f):
                if idx == greeting_line_idx:
                    greeting = json.loads(line)
                    greeting_text = greeting["text"]
                    greeting_audio = greeting["audio"]
                    break
        self.clear_history()
        return self.set_greeting(greeting_text, greeting_audio)

    def clear_history(self):
        self.history.clear()

    def read_wav(self, audio_path: str, sampling_rate: int):
        wav, raw_sample_rate = torchaudio.load(audio_path)  # (1, T)   tensor
        if raw_sample_rate != sampling_rate:
            wav = torchaudio.functional.resample(
                wav, raw_sample_rate, sampling_rate
            )  # tensor
        return wav

    def preprocess(
        self,
        task: Union[None, str] = None,
        input: Union[None, str] = None,
        instruction: Union[None, str] = None,
        add_silence_at_end=True,
        silence_frames=8,
        audio_channels=3,
        group_size=4,
        mode="s2s",
        transcript=None,
    ):
        if type(input) != str:  # speech input
            wav = (
                self.read_wav(input, self.generator.sampling_rate)
                .reshape(1, 1, -1)
                .to(self.device)
            )

            tokens = self.generator.inference_tokenize(wav)  # [n_vq, B, t]
            token_flat = (
                tokens.squeeze(1).permute(1, 0).reshape(-1).detach().cpu().numpy()
            )  # [T*n_q]

            silence_tokens = torch.tensor([688, 131, 226])
            token_flat = np.concatenate(
                [token_flat, np.tile(silence_tokens, silence_frames)]
            )
            token_flat = np.concatenate(
                [
                    token_flat,
                    np.tile(
                        silence_tokens,
                        (
                            group_size * audio_channels
                            - token_flat.shape[0] % (group_size * audio_channels)
                        )
                        // len(silence_tokens),
                    ),
                ]
            )
            audio_tokenized = torch.tensor(token_flat)
        else:
            text = input
            if (
                text.isupper() or text.islower()
            ):  # If the text only contains upper-case or lower-case letters, capitalize it.
                text = text.capitalize()
                
        assert self.greeting, "Must load greeting first"
                
        if task == 'tts':
            prompt = [
                InputSegment(f"[|Human|]: 请朗读这个句子: {text}"),
                InputSegment(" ###\n"),
                InputSegment(f"[|SpeechGPT|]: "),
            ]
            
        elif task == 'thought':

            if mode[0] == 's':  # s2t, s2s, st2t, st2s
                user_prompt = InputSegment("", audio=audio_tokenized)
                if mode[1] == 't':  # st2t, st2s
                    user_prompt += InputSegment(transcript)
                    system_prompt = InputSegment(
                        f"You are an helpful assistant. You should listen to the audio and answer the user's text questions in {'text' if mode[-1] == 't' else 'speech'}.\n\n\n",
                    )
                else:  # s2t, s2s
                    system_prompt = InputSegment(
                        f"You are an helpful assistant. You should answer the user's speech questions in {'text' if mode[-1] == 't' else 'speech'}.\n\n\n",
                    )
            elif mode[0] == 't':  # t2t, t2s
                user_prompt = InputSegment(text)
                system_prompt = InputSegment(
                    f"You are an helpful assistant. You should answer the user's text questions in {'text' if mode[-1] == 't' else 'speech'}.\n\n\n",
                )
                
            prompt = (
                [
                    system_prompt,
                    *self.greeting,
                ]
                if not self.history
                else []
            )
            prompt += [
                InputSegment(f"[|Human|]: "),
                user_prompt,
                InputSegment(f" ###\n[|SpeechGPT|]: "),
            ]

        input_ids = [seg.to_input_id(self.tokenizer, group_size) for seg in prompt]
        input_ids = torch.cat(input_ids, dim=1)

        return input_ids.to(self.device)

    def forward(
        self,
        task: Union[None, str] = None,
        input: Union[None, str] = None,
        instruction: Union[None, str] = None,
        mode: Union[None, str] = "s2s",
        text: Union[None, str] = None,
        audio_channels=3,
        max_new_tokens=5000,
    ):
        group_size = self.group_size
        with torch.no_grad():
            input_ids = self.preprocess(
                task=task,
                input=input,
                instruction=instruction,
                group_size=group_size,
                audio_channels=audio_channels,
                mode=mode,
                transcript=text,
            )

            generation_config = GenerationConfig(**self.generate_kwargs, max_new_tokens=max_new_tokens)

            input_ids = input_ids.T.reshape(1, -1)
            input_ids = torch.cat(self.history + [input_ids], dim=-1)
            prompt_length = input_ids.shape[1] // (audio_channels + 1)
            stopping_criteria = [
                MIMOStopper(
                    self.tokenizer.eos_token_id,
                    group_size,
                    audio_channels,
                    max_tokens=1024 + prompt_length,
                )
            ]

            generated_ids = self.model.generate(
                input_ids,
                generation_config,
                stopping_criteria=stopping_criteria,
            )
            # self.history.append(generated_ids)
            self.history = [generated_ids]

            generated_ids = (
                generated_ids.int().cpu().reshape(-1, 4).T[:, prompt_length:]
            )

            text = generated_ids[0, ::group_size][:-1]
            detokenized_text = self.tokenizer.decode(text, skip_special_tokens=True)

            answer = {
                "speech": "",
                "thought": detokenized_text,
                "result": "",
            }

            # Find <|sosp|> and <|eosp|> tokens locations in text channel token sequence
            sosp_idx_locations = (text == self.sosp_idx).nonzero(as_tuple=True)[0]
            eosp_idx_locations = (text == self.eosp_idx).nonzero(as_tuple=True)[0]
            if len(sosp_idx_locations) == 0:
                print("No <|sosp|> token found in the text channel")
            else:
                if len(eosp_idx_locations) == 0:
                    eosp_idx_locations = [text.shape[0]]
                sosp_idx_location = sosp_idx_locations[0] * group_size
                eosp_idx_location = eosp_idx_locations[0] * group_size
                audio_sequence = generated_ids[
                    :, sosp_idx_location + group_size : eosp_idx_location
                ]
                speech_sequence = audio_sequence[1:].T.flatten()
                assert (speech_sequence < 1024).all()
                answer["result"] = detokenized_text.strip().replace("<|empty|>", ".")

                answer["speech"] = "".join([f"<{i}>" for i in speech_sequence])

            # dump wav
            wav = torch.tensor(0)
            if answer["speech"]:
                tokens = torch.tensor(
                    [int(num) for num in re.findall(r"(\d+)>", answer["speech"])]
                )
                x = (
                    tokens.reshape(-1, 3)
                    .unsqueeze(0)
                    .permute(2, 0, 1)
                    .type(torch.LongTensor)
                    .to(self.device)
                )  # [n_vq, B, t]
                wav = self.generator.inference_detokenize(x)
                return detokenized_text, (24000, wav.reshape(-1).detach().cpu().numpy())

            return detokenized_text, None
        
    def get_ppl(self, inputs, mode):

        bsz = len(inputs)
        # params = self.model.params
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        with torch.no_grad():
            # tokenize
            input_tokens = [self.preprocess(
                input=x,
                group_size=self.group_size,
                task='thought',
                mode=mode,
            ) for x in inputs]
            max_prompt_size = max([t.shape[-1] for t in input_tokens])
            # total_len = min(params.max_seq_len, max_prompt_size)
            total_len = max_prompt_size
            tokens = torch.zeros((bsz, 4, total_len)).cuda().long()
            for k, t in enumerate(input_tokens):
                num_token = min(total_len, t.shape[-1])
                tokens[k, :, :num_token] = t[:, -num_token:].clone().long()
            # forward
            tokens = tokens.T.reshape(bsz, -1)
            outputs = self.model.forward(input_ids=tokens, labels=tokens)
            # compute ppl
            loss = outputs.loss
            loss = 10 * loss[0] + 4 * loss[1] + loss[2] + loss[3]
            return loss.item()