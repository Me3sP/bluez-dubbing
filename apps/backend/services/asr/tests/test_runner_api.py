import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import registry, runner_api  # noqa: E402
from common_schemas.models import ASRRequest, ASRResponse


@pytest.mark.parametrize("model_key", list(registry.WORKERS.keys()))
def test_call_worker_for_registered_models(monkeypatch, model_key):
    payload = ASRRequest(audio_url="dummy.wav")

    def fake_run(cmd, input, stdout, stderr, cwd, check, text):  # noqa: ANN001
        assert isinstance(cmd, list)
        return type(
            "Proc",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps({"segments": []}),
            },
        )()

    monkeypatch.setattr(runner_api, "UV_BIN", None, raising=False)
    monkeypatch.setattr(runner_api.subprocess, "run", fake_run)

    result = runner_api.call_worker(model_key, payload, ASRResponse, runner_index=0)
    assert isinstance(result, ASRResponse)
