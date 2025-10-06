from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import *
import httpx

ASR_URL = "http://localhost:8001/v1/transcribe"
TR_URL  = "http://localhost:8002/v1/translate"
TTS_URL = "http://localhost:8003/v1/synthesize"


app = FastAPI(title="orchestrator")

@app.post("/v1/dub")
async def dub(
    audio_url: str,
    target_lang: str,
    asr_model: str = Query("whisperx"),
    tr_model: str = Query("facebook_m2m100"),
    tts_model: str = Query("chatterbox"),
):
    async with httpx.AsyncClient(timeout=120) as client:
        a = await client.post(ASR_URL, params={"model_key": asr_model},
                              json=ASRRequest(audio_url=audio_url).model_dump())
        if a.status_code != 200: raise HTTPException(502, f"ASR failed: {a.text}")
        asr = ASRResponse(**a.json())

        t = await client.post(TR_URL, params={"model_key": tr_model},
                              json=TranslateRequest(segments=asr.segments, source_lang=asr.language, target_lang=target_lang).model_dump())
        if t.status_code != 200: raise HTTPException(502, f"Translate failed: {t.text}")
        tr = TranslateResponse(**t.json())

        translated_segments = []

        for translated_segment in tr.segments:
            translated_segments.append(
                SegmentAudioIn(
                    start=translated_segment.start,
                    end=translated_segment.end,
                    text=translated_segment.text,
                    speaker_id=translated_segment.speaker_id,
                    lang=translated_segment.lang or target_lang,
                    audio_prompt_url=audio_url
                )
            )


        s = await client.post(TTS_URL, params={"model_key": tts_model},
                              json=TTSRequest(segments=translated_segments).model_dump())
        if s.status_code != 200: raise HTTPException(502, f"TTS failed: {s.text}")
        tts = TTSResponse(**s.json())

    import uuid

    # Create unique workspace
    workspace_id = str(uuid.uuid4())
    BASE = Path(__file__).resolve().parents[4]
    output_dir = BASE / "outs" / workspace_id  # matches your repo structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save result to file
    output_file = output_dir / "final_result.json"
    with open(output_file, 'w') as f:
        f.write(tts.model_dump_json())

    return tts.model_dump_json()
