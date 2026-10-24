"""Placeholder model: repeats the prompt. Lets the whole pipeline run before a real model is chosen."""

from __future__ import annotations

from .base import LanguageModel


class EchoModel(LanguageModel):
    def generate(self, prompt: str) -> str:
        return f"[echo] {prompt}"
