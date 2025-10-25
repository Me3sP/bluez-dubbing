from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Dict, Any

# ASR
class ASRRequest(BaseModel):
    audio_url: str
    language_hint: str | None = None
    allow_short: bool = False # diarize after alignment when True
    # Free-form bucket for future-proofing
    extra: Dict[str, Any] = Field(default_factory=dict)

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
    lang: str | None = None
    audio_url: str | None = None  # Optional field for audio link to that segment

# will serve for both ASR and Translation responses
class ASRResponse(BaseModel):
    segments: List[Segment] = Field(default_factory=list)
    WordSegments: List[Word] | None = None
    language: str | None = None
    audio_url: str | None = None  # Optional field for audio link to that transcription
    # Free-form bucket for future-proofing
    extra: Dict[str, Any] = Field(default_factory=dict)


class ASRResultWrapper(BaseModel):
    raw: ASRResponse | None = None
    aligned: ASRResponse

# Translate
class TranslateRequest(BaseModel):
    segments: List[Segment] | None = Field(default_factory=list)
    source_lang: str | None = None
    target_lang: str
    # Free-form bucket for future-proofing
    extra: Dict[str, Any] = Field(default_factory=dict)


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
    audio_url: str | List[str]
    speaker_id: str | None = None
    lang: str | None = None
    sample_rate: int | None = None
class TTSRequest(BaseModel):
    segments: List[SegmentAudioIn] = Field(default_factory=list)
    workspace: str | None = None
    language: str | None = None
    # Free-form bucket for future-proofing
    extra: Dict[str, Any] = Field(default_factory=dict)
    

class TTSResponse(BaseModel):
    segments: List[SegmentAudioOut] = Field(default_factory=list)
    meta: Dict[str, Any] | None = None

class SubtitleSegment(BaseModel):
    """Represents a subtitle segment with timing and text."""
    start: float
    end: float
    text: str
    lines: List[str]
    
    @computed_field
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    @computed_field
    @property
    def cps(self) -> float:
        """Characters per second."""
        return self.char_count / self.duration if self.duration > 0 else 0