import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cli  # noqa: E402
from common_schemas.models import ASRRequest, ASRResponse, Segment


def test_build_parser_requires_audio(tmp_path):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_pipeline_run_invokes_both_runners(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    captured = []

    def fake_call_worker(model_key, payload, out_model, runner_index):  # noqa: ANN001
        captured.append((model_key, payload, runner_index))
        if runner_index == 0:
            return ASRResponse(
                segments=[Segment(text="hello")],
                language="en",
                audio_url=str(audio_file),
            )
        return ASRResponse(segments=[Segment(text="aligned hello")], language="en", audio_url=str(audio_file))

    monkeypatch.setattr(cli.runner_api, "call_worker", fake_call_worker)

    parser = cli.build_parser()
    args = parser.parse_args([str(audio_file), "--model-key", "test"])

    raw_result, aligned_result = cli.run(args)
    assert captured[0][2] == 0
    assert isinstance(captured[0][1], ASRRequest)
    assert captured[1][2] == 1
    assert isinstance(captured[1][1], ASRResponse)
    assert raw_result.segments[0].text == "hello"
    assert aligned_result.segments[0].text == "aligned hello"
    assert captured[1][1].extra["enable_diarization"] is True


def test_main_writes_json(monkeypatch, tmp_path, capsys):
    raw_result = ASRResponse(segments=[Segment(text="hello")])
    aligned_result = ASRResponse(segments=[Segment(text="aligned hello")])

    def fake_run(args):  # noqa: ANN001
        return raw_result, aligned_result

    monkeypatch.setattr(cli, "run", fake_run)

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    output_json = tmp_path / "result.json"
    raw_json = tmp_path / "raw.json"
    aligned_json = tmp_path / "aligned.json"

    exit_code = cli.main([str(audio_file), "--output-json", str(output_json)])

    assert exit_code == 0
    data = json.loads(output_json.read_text())
    assert data["raw"]["segments"][0]["text"] == "hello"
    assert data["aligned"]["segments"][0]["text"] == "aligned hello"
    stdout = capsys.readouterr().out
    assert "Combined ASR result saved" in stdout

    exit_code = cli.main([
        str(audio_file),
        "--raw-output-json",
        str(raw_json),
        "--aligned-output-json",
        str(aligned_json),
    ])
    assert exit_code == 0
    assert json.loads(raw_json.read_text())["segments"][0]["text"] == "hello"
    assert json.loads(aligned_json.read_text())["segments"][0]["text"] == "aligned hello"
