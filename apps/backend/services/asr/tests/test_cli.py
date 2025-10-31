import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cli  # noqa: E402
from common_schemas.models import ASRRequest, ASRResponse, ASRResultWrapper, Segment


def test_build_parser_requires_audio(tmp_path):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_run_invokes_call_worker(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    captured = {}

    def fake_call_worker(model_key, payload, out_model, runner_index):  # noqa: ANN001
        captured["model_key"] = model_key
        captured["payload"] = payload
        captured["runner_index"] = runner_index
        return ASRResultWrapper(
            raw=ASRResponse(segments=[]),
            aligned=ASRResponse(segments=[Segment(text="hello")]),
        )

    monkeypatch.setattr(cli.runner_api, "call_worker", fake_call_worker)

    parser = cli.build_parser()
    args = parser.parse_args([str(audio_file), "--model-key", "test", "--runner-index", "1", "--no-allow-short"])

    result = cli.run(args)
    assert isinstance(result, ASRResultWrapper)
    assert captured["model_key"] == "test"
    assert captured["runner_index"] == 1
    assert isinstance(captured["payload"], ASRRequest)
    assert captured["payload"].allow_short is False


def test_main_writes_json(monkeypatch, tmp_path, capsys):
    fake_result = ASRResultWrapper(
        raw=ASRResponse(segments=[]),
        aligned=ASRResponse(segments=[Segment(text="hello")]),
    )

    def fake_run(args):  # noqa: ANN001
        return fake_result

    monkeypatch.setattr(cli, "run", fake_run)

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    output_json = tmp_path / "result.json"
    exit_code = cli.main([str(audio_file), "--output-json", str(output_json)])

    assert exit_code == 0
    data = json.loads(output_json.read_text())
    assert data["aligned"]["segments"][0]["text"] == "hello"
    assert "saved" in capsys.readouterr().out
