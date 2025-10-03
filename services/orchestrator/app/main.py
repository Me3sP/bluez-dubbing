from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import *
import httpx

ASR_URL = "http://asr:8000/v1/transcribe"
TR_URL  = "http://translate:8000/v1/translate"
TTS_URL = "http://tts:8000/v1/synthesize"

app = FastAPI(title="orchestrator")

@app.post("/v1/dub")
async def dub(
    audio_url: str,
    target_lang: str,
    asr_model: str = Query("whisperx"),
    tr_model: str = Query("nllb"),
    tts_model: str = Query("xtts"),
):
    async with httpx.AsyncClient(timeout=120) as client:
        a = await client.post(ASR_URL, params={"model_key": asr_model},
                              json=ASRRequest(audio_url=audio_url).model_dump())
        if a.status_code != 200: raise HTTPException(502, f"ASR failed: {a.text}")
        asr = ASRResponse(**a.json())

        t = await client.post(TR_URL, params={"model_key": tr_model},
                              json=TranslateRequest(text=asr.text, target_lang=target_lang).model_dump())
        if t.status_code != 200: raise HTTPException(502, f"Translate failed: {t.text}")
        tr = TranslateResponse(**t.json())

        s = await client.post(TTS_URL, params={"model_key": tts_model},
                              json=TTSRequest(text=tr.text, lang=target_lang).model_dump())
        if s.status_code != 200: raise HTTPException(502, f"TTS failed: {s.text}")
        tts = TTSResponse(**s.json())

    return {"subtitle": tr.text, "dub_url": tts.audio_url}
