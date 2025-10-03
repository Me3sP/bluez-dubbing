from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import ASRRequest, ASRResponse
from .runner_api import call_worker

app = FastAPI(title="asr")

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/v1/transcribe", response_model=ASRResponse)
async def transcribe(req: ASRRequest, model_key: str = Query("whisperx")):
    try:
        return call_worker(model_key, req, ASRResponse)
    except Exception as e:
        raise HTTPException(500, str(e))
