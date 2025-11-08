from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


@lru_cache(maxsize=None)
def load_model_config(model_key: str) -> Dict[str, Any]:
    cfg = CONFIG_DIR / f"{model_key}.yaml"
    if not cfg.exists():
        raise RuntimeError(f"configuration file not found for model '{model_key}': {cfg}")
    return yaml.safe_load(cfg.read_text()) or {}


def read_model_languages(model_key: str) -> list[str]:
    data = load_model_config(model_key)
    langs = (
        data.get("languages")
        or data.get("supported_languages")
        or data.get("langs")
        or data.get("language_codes")
        or []
    )
    if isinstance(langs, dict):
        langs = list(langs.keys())
    return [str(lang) for lang in langs]


def get_service_logger(name: str, level: int, fmt: str = _LOG_FORMAT) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


@dataclass
class Worker:
    venv_python: Path
    runner: Path | Sequence[Path]
    languages: list[str]
