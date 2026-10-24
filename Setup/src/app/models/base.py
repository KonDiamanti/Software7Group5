"""Interface every language model provider must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LanguageModel(ABC):
    """Wrap any provider (hosted API, local model, ...) behind one method."""

    def __init__(self, name: str, temperature: float = 0.7, max_tokens: int = 1024):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's reply to the prompt."""
