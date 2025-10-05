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

class SegmentAudioIn(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str
    speaker_id: str | None = None
    lang: str | None = None
    audio_prompt_url: str | None = None

class SegmentAudioOut(BaseModel):
    start: float | None = None
    end: float | None = None
    audio_url: str
    speaker_id: str | None = None
    lang: str | None = None
    sample_rate: int | None = None
class TTSRequest(BaseModel):
    segments: List[SegmentAudioIn] = Field(default_factory=list)

class TTSResponse(BaseModel):
    Segments: List[SegmentAudioOut] = Field(default_factory=list)
    meta: Dict[str, Any] | None = None
