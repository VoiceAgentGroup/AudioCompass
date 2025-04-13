import torch
import time


class VoiceAssistant:
    @torch.no_grad()
    def generate_a2t(
        self,
        audio,
        max_new_tokens=2048,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_t2t(
        self,
        text,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_at2t(
        self,
        audio,
        text,
        max_new_tokens=2048,
    ):
        raise NotImplementedError
