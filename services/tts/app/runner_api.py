from __future__ import annotations
import json, subprocess
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel
from common_schemas.models import TTSRequest, TTSResponse
from .registry import get_worker  # maps model_key -> (venv_python, runner_path)

T = TypeVar("T", bound=BaseModel)

def _call_worker(venv_python: Path, runner: Path, payload: BaseModel, out_model: Type[T]) -> T:
    proc = subprocess.run(
        [str(venv_python), str(runner)],
        input=payload.model_dump_json().encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    return out_model(**json.loads(proc.stdout))

def synthesize(req: TTSRequest, model_key: str = "xtts") -> TTSResponse:
    vpy, runner = get_worker(model_key)
    return _call_worker(vpy, runner, req, TTSResponse)
