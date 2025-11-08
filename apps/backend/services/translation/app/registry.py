from __future__ import annotations
from pathlib import Path
from common_schemas.service_utils import Worker, read_model_languages

BASE = Path(__file__).resolve().parents[1]  # service root

# Map multiple models for this service
WORKERS = {
        "deep_translator": Worker(
            venv_python=BASE/"models/deepTranslationModel/.venv/bin/python",
            runner=BASE/"models/deepTranslationModel/runner.py",
            languages=read_model_languages("deep_translator"),
        ),
        "facebook_m2m100": Worker(
            venv_python=BASE/"models/facebook_m2m100Model/.venv/bin/python",
            runner=BASE/"models/facebook_m2m100Model/runner.py",
            languages=read_model_languages("facebook_m2m100"),
        ),
        
    }

def get_worker(model_key: str | None, source_language: int, target_language: str) -> tuple[Path, Path]:
    # Prefer the requested model if it supports the language
    selected_key = None
    if model_key in WORKERS:
        w = WORKERS[model_key]
        if w.languages and (source_language in w.languages and target_language in w.languages):
            selected_key = model_key

    # Otherwise, pick the first model that supports the language
    if selected_key is None:
        for k, w in WORKERS.items():
            if w.languages and (source_language in w.languages and target_language in w.languages):
                selected_key = k
                break

    # Fallback to the first model if none declare support (language unchanged)
    if selected_key is None:
        print(f"No model found supporting language={source_language} - {target_language}. Defaulting to first model.")
        selected_key = next(iter(WORKERS))

    w = WORKERS[selected_key]
    return w.venv_python, w.runner, selected_key
