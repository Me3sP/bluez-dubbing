from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import TTSRequest, TTSResponse
from .runner_api import call_worker

app = FastAPI(title="tts")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/v1/synthesize", response_model=TTSResponse)
async def tts_api(req: TTSRequest, model_key: str = Query("chatterbox", description="which TTS model to use")):
    try:
        return call_worker(model_key, req, TTSResponse)
    except Exception as e:
        raise HTTPException(500, str(e))
