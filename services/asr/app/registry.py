from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # service root

@dataclass
class Worker:
    venv_python: Path
    runner: Path

# Map *multiple* models for this service
WORKERS = {
    "whisperx": Worker(
        venv_python=BASE/"models/whisperxModel/.venv/bin/python",
        runner=[BASE/"models/whisperxModel/runner_0.py",
                BASE/"models/whisperxModel/runner_1.py"],
    ),
}

def get_worker(model_key: str, runner_index: int) -> tuple[Path, Path]:
    if model_key not in WORKERS:
        raise KeyError(f"Unknown model_key={model_key}, choose one of {list(WORKERS)}")
    w = WORKERS[model_key]
    return w.venv_python, w.runner[runner_index]
