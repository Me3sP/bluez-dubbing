from __future__ import annotations
from pathlib import Path
from common_schemas.service_utils import Worker, read_model_languages

BASE = Path(__file__).resolve().parents[1]  # service root

# Map multiple models for this service
WORKERS = {
    "whisperx": Worker(
        venv_python=BASE / "models/whisperxModel/.venv/bin/python",
        runner=[
            BASE / "models/whisperxModel/runner_0.py",
            BASE / "models/whisperxModel/runner_1.py",
        ],
        languages=read_model_languages("whisperx"),
    ),
}

def get_worker(model_key: str | None, runner_index: int, language: str) -> tuple[Path, Path]:
    # Prefer the requested model if it supports the language
    selected_key = None
    if model_key in WORKERS:
        w = WORKERS[model_key]
        if  w.languages and language in w.languages:
            selected_key = model_key

    # Otherwise, pick the first model that supports the language
    if selected_key is None:
        for k, w in WORKERS.items():
            if w.languages and language in w.languages:
                selected_key = k
                break

    # Fallback to the first model if none declare support (language unchanged)
    if selected_key is None:
        print(f"No model found supporting language={language}. Defaulting to first model.")
        selected_key = next(iter(WORKERS))

    w = WORKERS[selected_key]
    runners = w.runner if isinstance(w.runner, (list, tuple)) else [w.runner]
    idx = runner_index % len(runners)
    return w.venv_python, runners[idx], selected_key
