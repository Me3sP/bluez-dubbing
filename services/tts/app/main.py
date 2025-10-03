from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import TTSRequest, TTSResponse
from .runner_api import synthesize

app = FastAPI(title="tts")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/v1/synthesize", response_model=TTSResponse)
async def tts_api(req: TTSRequest, model_key: str = Query("xtts")):
    try:
        return synthesize(req, model_key=model_key)
    except Exception as e:
        raise HTTPException(500, str(e))
