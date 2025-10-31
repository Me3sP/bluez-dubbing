from __future__ import annotations
import json, subprocess, sys
import shutil
from pathlib import Path
from typing import Dict, Type, TypeVar
from pydantic import BaseModel
from .registry import get_worker
import yaml

T = TypeVar("T", bound=BaseModel)
BASE = Path(__file__).resolve().parents[1]  # service root
CONFIG_DIR = BASE.parent.parent / "libs/common-schemas/config"  # ../../libs/common-schemas/config
CONFIG_CACHE: Dict[str, Dict] = {}
UV_BIN = shutil.which("uv")


def _load_model_config(model_key: str) -> Dict:
    cfg = CONFIG_DIR / f"{model_key}.yaml"
    if not cfg.exists():
        raise RuntimeError(f"configuration file not found for model '{model_key}': {cfg}")
    if model_key not in CONFIG_CACHE:
        CONFIG_CACHE[model_key] = yaml.safe_load(cfg.read_text()) or {}
    return CONFIG_CACHE[model_key]


def call_worker(model_key: str, payload: BaseModel, out_model: type[T], runner_index: int) -> T:
    language = getattr(payload, "language_hint", None) if runner_index == 0 else getattr(payload, "language", None)
    venv_python, runner, selected_key = get_worker(model_key, runner_index, language)

    cfg = _load_model_config(selected_key)
    cfg_params = dict(cfg.get("params", {}))
    existing_extra = getattr(payload, "extra", {}) or {}
    merged_extra = {**cfg_params, **existing_extra}
    if hasattr(payload, "extra"):
        payload.extra = merged_extra

    cwd = runner.parent
    uv = UV_BIN
    cmd = [uv, "run", runner.name] if uv else [str(venv_python), str(runner)]

    proc = subprocess.run(
        cmd,
        input=payload.model_dump_json(),
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=str(cwd),
        check=False,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"worker failed ({proc.returncode}); see runner stderr for details.")
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError("worker produced no output. Check runner stderr for errors.")
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON from worker: {e}\nraw:\n{out}")
    return out_model(**data)
