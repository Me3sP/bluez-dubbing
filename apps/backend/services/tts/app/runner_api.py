from __future__ import annotations
import json
import subprocess
import sys
from typing import TypeVar

from pydantic import BaseModel

from .registry import get_worker  # maps model_key -> (venv_python, runner_path)
from common_schemas.service_utils import load_model_config

T = TypeVar("T", bound=BaseModel)


def call_worker(model_key: str, payload: BaseModel, out_model: type[T]) -> T:
    vpy, runner, selected_key = get_worker(model_key, payload.language)
    cfg = load_model_config(selected_key)
    cfg_params = dict(cfg.get("params", {}))
    existing_extra = getattr(payload, "extra", {}) or {}
    merged_extra = {**cfg_params, **existing_extra}
    payload.extra = merged_extra

    cmd = [str(vpy), str(runner)]
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
