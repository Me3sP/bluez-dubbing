from __future__ import annotations
import json, subprocess
import shutil
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel
from .registry import get_worker

T = TypeVar("T", bound=BaseModel)

def call_worker(model_key: str, payload: BaseModel, out_model: type[T], runner_index: int) -> T:
    venv_python, runner = get_worker(model_key, runner_index)
    cwd = runner.parent
    uv = shutil.which("uv")
    cmd = [uv, "run", runner.name] if uv else [str(venv_python), str(runner)]

    proc = subprocess.run(
        cmd,
        input=payload.model_dump_json().encode("utf-8"),
        capture_output=True,
        cwd=str(cwd),
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"worker failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'ignore')}")
    out = proc.stdout.decode("utf-8", "ignore").strip()
    if not out:
        raise RuntimeError(f"worker produced no output. stderr:\n{proc.stderr.decode('utf-8','ignore')}")
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON from worker: {e}\nraw:\n{out}\nstderr:\n{proc.stderr.decode('utf-8','ignore')}")
    return out_model(**data)
