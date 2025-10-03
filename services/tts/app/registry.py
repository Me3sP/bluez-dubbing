from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

BASE = Path(__file__).resolve().parents[1]

@dataclass
class Worker:
    venv_python: Path
    runner: Path

def get_worker(model_key: str) -> tuple[Path, Path]:
    # Map your models here
    workers = {
        "xtts": Worker(
            venv_python=BASE/"envs/xtts/.venv/bin/python",
            runner=BASE/"models/xtts/runner.py"
        ),
        "bark": Worker(
            venv_python=BASE/"envs/bark/.venv/bin/python",
            runner=BASE/"models/bark/runner.py"
        ),
    }
    if model_key not in workers:
        raise KeyError(f"Unknown TTS model: {model_key}")
    w = workers[model_key]
    return w.venv_python, w.runner
