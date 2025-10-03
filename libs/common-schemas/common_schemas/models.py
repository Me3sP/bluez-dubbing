from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ASR
class ASRRequest(BaseModel):
    audio_url: str
    language_hint: str | None = None

class Word(BaseModel):
    start: float
    end: float
    text: str
    score: float | None = None
    speaker_id: str | None = None

class Segment(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str
    words: List[Word] | None = None
    speaker_id: str | None = None

class ASRResponse(BaseModel):
    segments: List[Segment] = Field(default_factory=list)
    WordSegments: List[Word] | None = None
    language: str | None = None

# Translate
class TranslateRequest(BaseModel):
    text: str
    source_lang: str | None = None
    target_lang: str

class TranslateResponse(BaseModel):
    text: str

# TTS
class TTSRequest(BaseModel):
    text: str
    lang: str
    voice: str | None = None
    target_duration_s: float | None = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class TTSResponse(BaseModel):
    audio_url: str
    sample_rate: int
    meta: Dict[str, Any] = Field(default_factory=dict)
