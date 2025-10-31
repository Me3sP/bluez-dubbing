import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cli_1  # noqa: E402
from common_schemas.models import ASRResponse, Segment


def test_build_parser_requires_input():
    parser = cli_1.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_run_invokes_runner1(monkeypatch, tmp_path):
    payload = ASRResponse(
        segments=[Segment(text="hi")],
        language="en",
        audio_url="file.wav",
    )
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(payload.model_dump()), encoding="utf-8")

    captured = {}

    def fake_call_worker(model_key, payload, out_model, runner_index):  # noqa: ANN001
        captured["model_key"] = model_key
        captured["payload"] = payload
        captured["runner_index"] = runner_index
        return ASRResponse(segments=[Segment(text="aligned")])

    monkeypatch.setattr(cli_1.runner_api, "call_worker", fake_call_worker)

    parser = cli_1.build_parser()
    args = parser.parse_args([str(input_json), "--model-key", "test", "--no-diarize"])

    result = cli_1.run(args)

    assert isinstance(result, ASRResponse)
    assert captured["runner_index"] == 1
    assert isinstance(captured["payload"], ASRResponse)
    assert captured["payload"].extra["enable_diarization"] is False
