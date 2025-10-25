from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

BASE = Path(__file__).resolve().parents[1]  # service root
CONFIG_DIR = BASE.parent.parent / "libs/common-schemas/config"  # ../../libs/common-schemas/config

@dataclass
class Worker:
    venv_python: Path
    runner: list[Path]
    languages: list[str]

def read_model_languages(model_key: str) -> list[str]:
    cfg = CONFIG_DIR / f"{model_key}.yaml"
    if not cfg.exists():
        raise RuntimeError(f"configuration file not found for model '{model_key}': {cfg}")
    
    data = yaml.safe_load(cfg.read_text()) or {}
    langs = (
        data.get("languages")
        or data.get("supported_languages")
        or data.get("langs")
        or data.get("language_codes")
        or []
    )
    if isinstance(langs, dict):
        langs = list(langs.keys())
    return [str(x) for x in langs]

# Map multiple models for this service
WORKERS = {
        "chatterbox": Worker(
            venv_python=BASE/"models/chatterboxModel/.venv/bin/python",
            runner=BASE/"models/chatterboxModel/runner.py",
            languages=read_model_languages("chatterbox")
        ),
        "edge_tts": Worker(
            venv_python=BASE/"models/edgeTTsModel/.venv/bin/python",
            runner=BASE/"models/edgeTTsModel/runner.py",
            languages=read_model_languages("edge_tts")
        ),
    }

def get_worker(model_key: str | None, language: str) -> tuple[Path, Path]:
    # Prefer the requested model if it supports the language
    selected_key = None
    if model_key in WORKERS:
        w = WORKERS[model_key]
        if not w.languages or language in w.languages:
            selected_key = model_key

    # Otherwise, pick the first model that supports the language
    if selected_key is None:
        for k, w in WORKERS.items():
            if not w.languages or language in w.languages:
                selected_key = k
                break

    # Fallback to the first model if none declare support (language unchanged)
    if selected_key is None:
        print(f"No model found supporting language={language}. Defaulting to first model.")
        selected_key = next(iter(WORKERS))

    w = WORKERS[selected_key]
    return w.venv_python, w.runner, selected_key