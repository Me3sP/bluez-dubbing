from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import ASRRequest, ASRResponse
from .runner_api import call_worker
from typing import Union

app = FastAPI(title="asr")

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/v1/transcribe", response_model=ASRResponse)
async def transcribe(
    req: Union[ASRRequest, ASRResponse],
    model_key: str = Query("whisperx"),
    runner_index: int = Query(0, ge=0, le=1),
    diarize: bool = Query(True),
):
    try:
        if runner_index == 0:
            if not isinstance(req, ASRRequest):
                raise HTTPException(400, "Runner 0 expects an ASRRequest payload.")
            return call_worker(model_key, req, ASRResponse, runner_index)

        if not isinstance(req, ASRResponse):
            raise HTTPException(400, "Runner 1 expects an ASRResponse payload.")
        req.extra = dict(req.extra or {})
        req.extra["enable_diarization"] = diarize
        return call_worker(model_key, req, ASRResponse, runner_index)
    except Exception as e:
        raise HTTPException(500, str(e))
