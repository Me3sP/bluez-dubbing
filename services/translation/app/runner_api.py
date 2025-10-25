from __future__ import annotations
import json, subprocess
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel
from .registry import get_worker  # maps model_key -> (venv_python, runner_path)
from shutil import which
import yaml

T = TypeVar("T", bound=BaseModel)
BASE = Path(__file__).resolve().parents[1]  # service root
CONFIG_DIR = BASE.parent.parent / "libs/common-schemas/config"  # ../../libs/common-schemas/config

def call_worker(model_key: str, payload: BaseModel, out_model: type[T]) -> T:

    vpy, runner, selected_key = get_worker(model_key, payload.source_lang, payload.target_lang)
    cfg = CONFIG_DIR / f"{selected_key}.yaml"
    data = yaml.safe_load(cfg.read_text()) or {}
    payload.extra = data.get("params", {})

    uv = which("uv")
    cmd = [uv, "run", runner.name] if uv else [str(vpy), str(runner)]
    cwd = runner.parent
    proc = subprocess.run(
        cmd,
        input=payload.model_dump_json().encode("utf-8"),
        capture_output=True,
        cwd=cwd,
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
