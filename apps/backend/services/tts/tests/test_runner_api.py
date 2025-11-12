import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import registry, runner_api  # noqa: E402
from common_schemas.models import SegmentAudioIn, TTSRequest, TTSResponse


@pytest.mark.parametrize("model_key", list(registry.WORKERS.keys()))
def test_tts_call_worker(monkeypatch, model_key):
    request = TTSRequest(segments=[SegmentAudioIn(text="hello")], workspace="/tmp", language="en")

    def fake_run(*_args, **_kwargs):  # noqa: ANN001
        return type(
            "Proc",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps({"segments": []}).encode("utf-8"),
                "stderr": b"",
            },
        )()

    monkeypatch.setattr(runner_api.subprocess, "run", fake_run)

    result = runner_api.call_worker(model_key, request, TTSResponse)
    assert isinstance(result, TTSResponse)
