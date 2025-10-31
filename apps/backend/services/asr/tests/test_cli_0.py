import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cli_0  # noqa: E402
from common_schemas.models import ASRRequest, ASRResponse, Segment


def test_build_parser_requires_audio():
    parser = cli_0.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_run_invokes_runner0(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    captured = {}

    def fake_call_worker(model_key, payload, out_model, runner_index):  # noqa: ANN001
        captured["model_key"] = model_key
        captured["payload"] = payload
        captured["runner_index"] = runner_index
        return ASRResponse(segments=[Segment(text="hello")])

    monkeypatch.setattr(cli_0.runner_api, "call_worker", fake_call_worker)

    parser = cli_0.build_parser()
    args = parser.parse_args([str(audio_file), "--model-key", "test", "--min-speakers", "1"])
    result = cli_0.run(args)

    assert isinstance(result, ASRResponse)
    assert captured["runner_index"] == 0
    assert isinstance(captured["payload"], ASRRequest)
    assert captured["payload"].min_speakers == 1
