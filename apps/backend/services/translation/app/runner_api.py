from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, TypeVar

from pydantic import BaseModel

from .registry import get_worker  # maps model_key -> (venv_python, runner_path)
from shutil import which
import yaml

T = TypeVar("T", bound=BaseModel)
BASE = Path(__file__).resolve().parents[1]  # service root
CONFIG_DIR = BASE.parent.parent / "libs/common-schemas/config"  # ../../libs/common-schemas/config
CONFIG_CACHE: Dict[str, Dict] = {}
UV_BIN = which("uv")


def _load_model_config(model_key: str) -> Dict:
    cfg = CONFIG_DIR / f"{model_key}.yaml"
    if not cfg.exists():
        raise RuntimeError(f"configuration file not found for model '{model_key}': {cfg}")
    if model_key not in CONFIG_CACHE:
        CONFIG_CACHE[model_key] = yaml.safe_load(cfg.read_text()) or {}
    return CONFIG_CACHE[model_key]


def _format_cmd(venv_python: Path, runner: Path) -> Tuple[str, ...]:
    if UV_BIN:
        return (UV_BIN, "run", runner.name)
    return (str(venv_python), str(runner))


def call_worker(model_key: str, payload: BaseModel, out_model: type[T]) -> T:
    vpy, runner, selected_key = get_worker(model_key, payload.source_lang, payload.target_lang)
    cfg = _load_model_config(selected_key)
    cfg_params = dict(cfg.get("params", {}))
    existing_extra = getattr(payload, "extra", {}) or {}
    merged_extra = {**cfg_params, **existing_extra}
    payload.extra = merged_extra

    cmd = list(_format_cmd(vpy, runner))
    proc = subprocess.run(
        cmd,
        input=payload.model_dump_json(),
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=runner.parent,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed ({proc.returncode}); see runner stderr for details.")
    out = (proc.stdout or "").strip()

    if not out:
        raise RuntimeError("worker produced no output")
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON from worker: {e}\nraw:\n{out}")
    return out_model(**data)
