import asyncio
import sys
import wave
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import main as orchestrator_main  # noqa: E402
from common_schemas.models import ASRResponse, Segment, SegmentAudioOut, TTSResponse


@pytest.mark.asyncio
async def test_dub_pipeline_minimal(monkeypatch, tmp_path):
    # Create a 1-second silent wav input to avoid ffmpeg dependency.
    input_wav = tmp_path / "input.wav"
    with wave.open(str(input_wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)

    original_outs = orchestrator_main.OUTS
    orchestrator_main.OUTS = tmp_path

    async def fake_maybe_run_audio_separation(*args, **kwargs):  # noqa: ANN001
        return None, None, "default"

    async def fake_run_asr_step(*args, **kwargs):  # noqa: ANN001
        seg = Segment(start=0.0, end=1.0, text="Hello", speaker_id="spk1", lang="en")
        response = ASRResponse(segments=[seg], language="en")
        return response, response

    async def fake_run_translation_step(*args, **kwargs):  # noqa: ANN001
        seg = Segment(start=0.0, end=1.0, text="Bonjour", speaker_id="spk1", lang="fr")
        return ASRResponse(segments=[seg], language="fr")

    tts_audio = tmp_path / "tts.wav"
    with wave.open(str(tts_audio), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)

    async def fake_synthesize_tts(*args, **kwargs):  # noqa: ANN001
        segment_out = SegmentAudioOut(
            start=0.0,
            end=1.0,
            audio_url=str(tts_audio),
            speaker_id="spk1",
            lang="fr",
            sample_rate=16000,
        )
        return TTSResponse(segments=[segment_out])

    async def fake_trim_tts_segments(tts_result, vad_dir):  # noqa: ANN001
        return tts_result

    async def fake_concatenate_segments(*args, **kwargs):  # noqa: ANN001
        speech_track = tmp_path / "speech.wav"
        speech_track.write_bytes(tts_audio.read_bytes())
        return str(speech_track), [{"start": 0.0, "end": 1.0, "text": "Bonjour", "speaker_id": "spk1"}]

    async def fake_align_dubbed_audio(*args, **kwargs):  # noqa: ANN001
        seg = Segment(start=0.0, end=1.0, text="Bonjour", speaker_id="spk1", lang="fr")
        return ASRResponse(segments=[seg], language="fr")

    async def fake_finalize_media(
        video_path,
        audio_path,
        dubbed_path,
        output_path,
        subtitle_path,
        sub_style,
        mobile_optimized,
        dubbing_strategy,
    ):  # noqa: ANN001
        Path(dubbed_path).write_text("video")
        if output_path:
            Path(output_path).write_text("video with subtitles")

    monkeypatch.setattr(orchestrator_main, "maybe_run_audio_separation", fake_maybe_run_audio_separation)
    monkeypatch.setattr(orchestrator_main, "run_asr_step", fake_run_asr_step)
    monkeypatch.setattr(orchestrator_main, "run_translation_step", fake_run_translation_step)
    monkeypatch.setattr(orchestrator_main, "synthesize_tts", fake_synthesize_tts)
    monkeypatch.setattr(orchestrator_main, "trim_tts_segments", fake_trim_tts_segments)
    monkeypatch.setattr(orchestrator_main, "concatenate_segments", fake_concatenate_segments)
    monkeypatch.setattr(orchestrator_main, "align_dubbed_audio", fake_align_dubbed_audio)
    monkeypatch.setattr(orchestrator_main, "finalize_media", fake_finalize_media)
    monkeypatch.setattr(orchestrator_main, "get_audio_duration", lambda *args, **kwargs: 1.0)  # noqa: ARG005

    await orchestrator_main.startup_event()
    try:
        result = await orchestrator_main.dub(
            video_url=str(input_wav),
            target_work="dub",
            target_langs=["fr"],
            source_lang="en",
            translation_strategy="default",
            dubbing_strategy="default",
            sophisticated_dub_timing=True,
            subtitle_style="default_mobile",
            audio_sep=False,
            perform_vad_trimming=False,
            persist_intermediate=False,
            sep_model="melband_roformer_big_beta5e.ckpt",
            asr_model="whisperx",
            tr_model="facebook_m2m100",
            tts_model="chatterbox",
            run_id=None,
            involve_mode=False,
        )
    finally:
        await orchestrator_main.shutdown_event()
        orchestrator_main.OUTS = original_outs

    assert result["final_video_path"]
    assert result["workspace_id"]
    assert result.get("default_language") == "fr"
    assert "fr" in (result.get("available_languages") or [])
    assert result.get("language_outputs", {}).get("fr", {}).get("final_video_path")


@pytest.mark.asyncio
async def test_dub_pipeline_multiple_languages(monkeypatch, tmp_path):
    input_wav = tmp_path / "input.wav"
    with wave.open(str(input_wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)

    original_outs = orchestrator_main.OUTS
    orchestrator_main.OUTS = tmp_path

    async def fake_maybe_run_audio_separation(*args, **kwargs):  # noqa: ANN001
        return None, None, "default"

    async def fake_run_asr_step(*args, **kwargs):  # noqa: ANN001
        seg = Segment(start=0.0, end=1.0, text="Hello", speaker_id="spk1", lang="en")
        response = ASRResponse(segments=[seg], language="en")
        return response, response

    translations = {"fr": "Bonjour", "es": "Hola"}

    async def fake_run_translation_step(*args, **kwargs):  # noqa: ANN001
        target = args[4] if len(args) > 4 else kwargs.get("target_lang", "fr")
        text = translations.get(target, "Hello")
        seg = Segment(start=0.0, end=1.0, text=text, speaker_id="spk1", lang=target)
        return ASRResponse(segments=[seg], language=target)

    tts_audio = tmp_path / "tts_multi.wav"
    with wave.open(str(tts_audio), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)

    async def fake_synthesize_tts(*args, **kwargs):  # noqa: ANN001
        target_lang = args[3] if len(args) > 3 else kwargs.get("target_lang", "fr")
        segment_out = SegmentAudioOut(
            start=0.0,
            end=1.0,
            audio_url=str(tts_audio),
            speaker_id="spk1",
            lang=target_lang,
            sample_rate=16000,
        )
        return TTSResponse(segments=[segment_out])

    async def fake_trim_tts_segments(tts_result, vad_dir):  # noqa: ANN001
        return tts_result

    async def fake_concatenate_segments(*args, **kwargs):  # noqa: ANN001
        lang = "multi"
        translation_segments = kwargs.get("translation_segments") or []
        if translation_segments:
            lang = translation_segments[0].get("lang", "multi")
        speech_track = tmp_path / f"speech_{lang}.wav"
        speech_track.write_bytes(tts_audio.read_bytes())
        return str(speech_track), translation_segments or [{"start": 0.0, "end": 1.0, "text": translations.get(lang, "Hello"), "speaker_id": "spk1", "lang": lang}]

    async def fake_align_dubbed_audio(*args, **kwargs):  # noqa: ANN001
        tr_result = args[2] if len(args) > 2 else kwargs.get("tr_result")
        return tr_result or ASRResponse(segments=[])

    async def fake_finalize_media(
        video_path,
        audio_path,
        dubbed_path,
        output_path,
        subtitle_path,
        sub_style,
        mobile_optimized,
        dubbing_strategy,
    ):  # noqa: ANN001
        Path(dubbed_path).write_text("video")
        if output_path:
            Path(output_path).write_text("video with subtitles")

    monkeypatch.setattr(orchestrator_main, "maybe_run_audio_separation", fake_maybe_run_audio_separation)
    monkeypatch.setattr(orchestrator_main, "run_asr_step", fake_run_asr_step)
    monkeypatch.setattr(orchestrator_main, "run_translation_step", fake_run_translation_step)
    monkeypatch.setattr(orchestrator_main, "synthesize_tts", fake_synthesize_tts)
    monkeypatch.setattr(orchestrator_main, "trim_tts_segments", fake_trim_tts_segments)
    monkeypatch.setattr(orchestrator_main, "concatenate_segments", fake_concatenate_segments)
    monkeypatch.setattr(orchestrator_main, "align_dubbed_audio", fake_align_dubbed_audio)
    monkeypatch.setattr(orchestrator_main, "finalize_media", fake_finalize_media)
    monkeypatch.setattr(orchestrator_main, "get_audio_duration", lambda *args, **kwargs: 1.0)  # noqa: ARG005

    await orchestrator_main.startup_event()
    try:
        result = await orchestrator_main.dub(
            video_url=str(input_wav),
            target_work="dub",
            target_langs=["fr", "es"],
            source_lang="en",
            translation_strategy="default",
            dubbing_strategy="default",
            sophisticated_dub_timing=True,
            subtitle_style="default",
            audio_sep=False,
            perform_vad_trimming=False,
            persist_intermediate=False,
            sep_model="melband_roformer_big_beta5e.ckpt",
            asr_model="whisperx",
            tr_model="facebook_m2m100",
            tts_model="chatterbox",
            run_id=None,
            involve_mode=False,
        )
    finally:
        await orchestrator_main.shutdown_event()
        orchestrator_main.OUTS = original_outs

    language_outputs = result.get("language_outputs", {})
    assert set(result.get("available_languages") or []) == {"fr", "es"}
    assert result.get("default_language") == "fr"
    assert language_outputs.get("es", {}).get("final_video_path")
