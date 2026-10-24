"""Guard model registry.

To add a guard: subclass Guard, then add it to REGISTRY below.
"""

from __future__ import annotations

from ..config import GuardSettings
from .allow_all import AllowAllGuard
from .base import Direction, Guard, GuardResult

REGISTRY: dict[str, type[Guard]] = {
    "allow_all": AllowAllGuard,
    # "my_guard": MyGuard,   # TBD once the guard model is chosen
}


def build_guard(settings: GuardSettings) -> Guard:
    try:
        cls = REGISTRY[settings.provider]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown guard provider '{settings.provider}'. Registered: {known}"
        ) from None
    return cls(name=settings.name)


__all__ = ["Guard", "GuardResult", "Direction", "AllowAllGuard", "REGISTRY", "build_guard"]
