import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cli  # noqa: E402
from common_schemas.models import SegmentAudioOut, TTSResponse


def test_parser_requires_input(tmp_path):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_run_invokes_worker(monkeypatch, tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"segments": [{"text": "Bonjour"}]}))

    workspace = tmp_path / "workspace"

    captured = {}

    def fake_call_worker(model_key, request, out_model):  # noqa: ANN001
        captured["model_key"] = model_key
        captured["request"] = request
        return TTSResponse(segments=[SegmentAudioOut(audio_url="file.wav")])

    monkeypatch.setattr(cli.runner_api, "call_worker", fake_call_worker)

    parser = cli.build_parser()
    args = parser.parse_args([str(input_file), "--workspace", str(workspace), "--model-key", "edge", "--language", "fr"])

    result = cli.run(args)

    assert isinstance(result, TTSResponse)
    assert captured["model_key"] == "edge"
    assert captured["request"].language == "fr"
    assert captured["request"].workspace == str(workspace.resolve())


def test_main_outputs_json(monkeypatch, tmp_path, capsys):
    output_resp = TTSResponse(segments=[SegmentAudioOut(audio_url="seg-0.wav")])

    def fake_run(args):  # noqa: ANN001
        return output_resp

    monkeypatch.setattr(cli, "run", fake_run)

    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"segments": []}))

    output_path = tmp_path / "tts.json"
    exit_code = cli.main([str(input_file), "--output-json", str(output_path)])

    assert exit_code == 0
    assert json.loads(output_path.read_text())["segments"][0]["audio_url"] == "seg-0.wav"
    assert "TTS result saved" in capsys.readouterr().out
