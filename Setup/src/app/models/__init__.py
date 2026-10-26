"""Language model registry.

To add a provider: subclass LanguageModel, then add it to REGISTRY below.
"""

from __future__ import annotations

from ..config import ModelSettings
from .base import LanguageModel
from .echo import EchoModel

REGISTRY: dict[str, type[LanguageModel]] = {
    "echo": EchoModel,
    # "my_provider": MyProviderModel,   # TBD once the tier is chosen
}


def build_model(settings: ModelSettings) -> LanguageModel:
    try:
        cls = REGISTRY[settings.provider]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown model provider '{settings.provider}'. Registered: {known}"
        ) from None
    return cls(
        name=settings.name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


__all__ = ["LanguageModel", "EchoModel", "REGISTRY", "build_model"]
