from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import TranslateRequest, ASRResponse
from .runner_api import call_worker

app = FastAPI(title="translation service", version="0.1.0")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/v1/translate", response_model=ASRResponse)
async def translate_api(req: TranslateRequest, model_key: str = Query("facebook_m2m100", description="which translation model to use")):
    try:
        return call_worker(model_key, req, ASRResponse)
    except Exception as e:
        raise HTTPException(500, str(e))
