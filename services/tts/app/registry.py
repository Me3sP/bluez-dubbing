from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

BASE = Path(__file__).resolve().parents[1]

@dataclass
class Worker:
    venv_python: Path
    runner: Path

WORKERS = {
        "chatterbox": Worker(
            venv_python=BASE/"models/chatterboxModel/.venv/bin/python",
            runner=BASE/"models/chatterboxModel/runner.py"
        ),
        "edge_tts": Worker(
            venv_python=BASE/"models/edgeTTsModel/.venv/bin/python",
            runner=BASE/"models/edgeTTsModel/runner.py"
        ),
    }

def get_worker(model_key: str) -> tuple[Path, Path]:

    if model_key not in WORKERS:
        raise KeyError(f"Unknown model_key={model_key}, choose one of {list(WORKERS)}")
    w = WORKERS[model_key]
    return w.venv_python, w.runner
