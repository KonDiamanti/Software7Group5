"""Load settings from config/settings.yaml, with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class ModelSettings:
    provider: str = "echo"
    name: str = "placeholder"
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass(frozen=True)
class GuardSettings:
    provider: str = "allow_all"
    name: str = "placeholder"
    check_input: bool = True
    check_output: bool = True


@dataclass(frozen=True)
class Settings:
    model: ModelSettings
    guard: GuardSettings
    database_path: Path


def load_settings(path: Path | str | None = None) -> Settings:
    """Read settings.yaml. Missing keys fall back to the placeholder defaults."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    raw: dict = {}
    if settings_path.exists():
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}

    db_path = os.environ.get("DATABASE_PATH") or raw.get("database", {}).get(
        "path", "data/app.db"
    )
    db_path = Path(db_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    return Settings(
        model=ModelSettings(**raw.get("model", {})),
        guard=GuardSettings(**raw.get("guard", {})),
        database_path=db_path,
    )
