import torch
import time


class VoiceAssistant:
    @torch.no_grad()
    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_text(
        self,
        text,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_mixed(
        self,
        audio,
        text,
        max_new_tokens=2048,
    ):
        raise NotImplementedError
