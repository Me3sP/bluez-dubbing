import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cli  # noqa: E402
from common_schemas.models import ASRResponse, Segment


def test_parser_requires_target_lang(tmp_path):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([str(tmp_path / "input.json")])


def test_run_invokes_worker(monkeypatch, tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"segments": [{"text": "hello"}]}))

    captured = {}

    def fake_call_worker(model_key, request, out_model):  # noqa: ANN001
        captured["model_key"] = model_key
        captured["request"] = request
        return ASRResponse(segments=[Segment(text="bonjour")], language="fr")

    monkeypatch.setattr(cli.runner_api, "call_worker", fake_call_worker)

    parser = cli.build_parser()
    args = parser.parse_args([str(input_file), "--target-lang", "fr", "--model-key", "m2m", "--source-lang", "en"])

    result = cli.run(args)

    assert isinstance(result, ASRResponse)
    assert captured["model_key"] == "m2m"
    assert captured["request"].source_lang == "en"
    assert captured["request"].target_lang == "fr"


def test_main_outputs_json(monkeypatch, tmp_path, capsys):
    output_data = ASRResponse(segments=[Segment(text="hola")])

    def fake_run(args):  # noqa: ANN001
        return output_data

    monkeypatch.setattr(cli, "run", fake_run)

    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"segments": []}))

    output_path = tmp_path / "translation.json"
    exit_code = cli.main([str(input_file), "--target-lang", "es", "--output-json", str(output_path)])

    assert exit_code == 0
    assert json.loads(output_path.read_text())["segments"][0]["text"] == "hola"
    assert "Translation saved" in capsys.readouterr().out
