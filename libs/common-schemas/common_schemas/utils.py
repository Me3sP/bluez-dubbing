from .models import Word, Segment


# ---- Convert WhisperX result to ASRResponse ----
def to_word(w: dict, seg_speaker: str | None) -> Word:
    # words can use "word" or "text" as the token key
    text = (w.get("word") or w.get("text") or "").strip()
    start = w.get("start")
    end = w.get("end", start)
    speaker = w.get("speaker") or seg_speaker
    score = w.get("score")
    return Word(
        start=float(start) if start is not None else 0.0,
        end=float(end) if end is not None else 0.0,
        text=text,
        score=score,
        speaker_id=speaker,
    )

def convert_whisperx_result_to_Segment(result: dict) -> Segment:
    segments_out: list[Segment] = []
    for seg in result.get("segments", []):
        seg_speaker = seg.get("speaker")
        words_list = [to_word(w, seg_speaker) for w in seg.get("words", [])]
        segments_out.append(
            Segment(
                start=float(seg["start"]) if seg.get("start") is not None else None,
                end=float(seg["end"]) if seg.get("end") is not None else None,
                text=(seg.get("text") or "").strip(),
                words=words_list or None,
                speaker_id=seg_speaker,
            )
        )

    return segments_out

def create_word_segments(result: dict, segments_out: list[Segment]) -> list[Word]:
    # Prefer explicit word_segments from WhisperX; otherwise, flatten from segments
    word_segments_out: list[Word] = []
    raw_word_segments = result.get("word_segments")
    if isinstance(raw_word_segments, list) and raw_word_segments:
        for w in raw_word_segments:
            word_segments_out.append(to_word(w, seg_speaker=None))
    else:
        for s in segments_out:
            if s.words:
                word_segments_out.extend(s.words)

    return word_segments_out

