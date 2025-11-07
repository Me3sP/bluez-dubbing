from __future__ import annotations
import json
import os
import subprocess
import threading
import sys
from pathlib import Path
from typing import Dict, Tuple, Type, TypeVar

from pydantic import BaseModel

from .registry import get_worker  # maps model_key -> (venv_python, runner_path)
import yaml

T = TypeVar("T", bound=BaseModel)
BASE = Path(__file__).resolve().parents[1]  # service root
CONFIG_DIR = BASE.parent.parent / "libs/common-schemas/config"  # ../../libs/common-schemas/config
CONFIG_CACHE: Dict[str, Dict] = {}
USE_PERSISTENT = os.getenv("USE_PERSISTENT_WORKERS", "0") == "1"
PROCESS_POOL: Dict[Tuple[Tuple[str, ...], str], "PersistentWorker"] = {}
POOL_LOCK = threading.Lock()


class PersistentWorker:
    def __init__(self, cmd: Tuple[str, ...], cwd: Path, env: Dict[str, str] | None = None):
        self._cmd = list(cmd)
        self._cwd = str(cwd)
        self._base_env = env or {}
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._spawn()

    def _spawn(self) -> None:
        env = os.environ.copy()
        env.update(self._base_env)
        env.pop("VIRTUAL_ENV", None)
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                cwd=self._cwd,
                env=env,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Failed to spawn worker command: {self._cmd}") from exc

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
        return self._proc  # type: ignore[return-value]

    def run(self, payload: str) -> str:
        with self._lock:
            proc = self._ensure_proc()
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
                response = proc.stdout.readline()
            except (BrokenPipeError, OSError):
                self._spawn()
                proc = self._ensure_proc()
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
                response = proc.stdout.readline()
            if not response:
                raise RuntimeError("Persistent worker produced no output.")
            return response.strip()


def _load_model_config(model_key: str) -> Dict:
    cfg = CONFIG_DIR / f"{model_key}.yaml"
    if not cfg.exists():
        raise RuntimeError(f"configuration file not found for model '{model_key}': {cfg}")
    if model_key not in CONFIG_CACHE:
        CONFIG_CACHE[model_key] = yaml.safe_load(cfg.read_text()) or {}
    return CONFIG_CACHE[model_key]


def _get_persistent_worker(cmd: Tuple[str, ...], cwd: Path) -> PersistentWorker:
    key = (cmd, str(cwd))
    with POOL_LOCK:
        worker = PROCESS_POOL.get(key)
        if worker is None:
            worker = PersistentWorker(cmd, cwd, env={"PERSISTENT_WORKER": "1"})
            PROCESS_POOL[key] = worker
    return worker


def call_worker(model_key: str, payload: BaseModel, out_model: type[T]) -> T:
    vpy, runner, selected_key = get_worker(model_key, payload.language)
    cfg = _load_model_config(selected_key)
    cfg_params = dict(cfg.get("params", {}))
    existing_extra = getattr(payload, "extra", {}) or {}
    merged_extra = {**cfg_params, **existing_extra}
    payload.extra = merged_extra

    if not USE_PERSISTENT:
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
    else:
        cmd = (str(vpy), str(runner))
        worker = _get_persistent_worker(cmd, runner.parent)
        out = worker.run(payload.model_dump_json())

    if not out:
        raise RuntimeError("worker produced no output")
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON from worker: {e}\nraw:\n{out}")
    return out_model(**data)
