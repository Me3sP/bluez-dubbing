from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import ASRRequest, ASRResponse
from .runner_api import call_worker
from typing import Union

app = FastAPI(title="asr")

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/v1/transcribe", response_model=ASRResponse)
async def transcribe(req: Union[ASRRequest, ASRResponse], model_key: str = Query("whisperx"), runner_index: int = Query(0, ge=0, le=1)):
    try:
        return call_worker(model_key, req, ASRResponse, runner_index)
    except Exception as e:
        raise HTTPException(500, str(e))
