from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

BASE = Path(__file__).resolve().parents[1]

@dataclass
class Worker:
    venv_python: Path
    runner: Path

WORKERS = {
        "facebook_m2m100": Worker(
            venv_python=BASE/"models/facebook_m2m100Model/.venv/bin/python",
            runner=BASE/"models/facebook_m2m100Model/runner.py"
        ),
        "deep_translator": Worker(
            venv_python=BASE/"models/deepTranslationModel/.venv/bin/python",
            runner=BASE/"models/deepTranslationModel/runner.py"
        ),
    }

def get_worker(model_key: str) -> tuple[Path, Path]:

    if model_key not in WORKERS:
        print(f"Unknown model_key={model_key}, choose one of {list(WORKERS)}")
        print("Defaulting to first available model.")
        model_key = list(WORKERS)[0]  # default to first
        
    w = WORKERS[model_key]
    return w.venv_python, w.runner
