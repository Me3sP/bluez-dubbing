from .models import Word, Segment, ASRResponse
from typing import List, Tuple, Optional, Set
from simalign import SentenceAligner
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Set 
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import soundfile as sf
import math

DETERMINERS = {
    # English
    'en': {
        'the', 'a', 'an',
        'this', 'that', 'these', 'those',
        'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'some', 'any', 'each', 'every', 'no', 'none', 'few', 'many', 'much', 'several', 'either', 'neither', 'all', 'both'
    },

    # 🇫🇷 French
    'fr': {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 
        'ce', 'cet', 'cette', 'ces',
        'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
        'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
        'aucun', 'chaque', 'plusieurs', 'quelques', 'tous', 'tout', 'toute', 'toutes'
    },

    # 🇪🇸 Spanish
    'es': {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'del', 'al',
        'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas',
        'mi', 'mis', 'tu', 'tus', 'su', 'sus',
        'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras',
        'ningún', 'ninguna', 'cada', 'algunos', 'algunas', 'todos', 'todas'
    },

    # 🇩🇪 German
    'de': {
        'der', 'die', 'das', 'den', 'dem', 'des',
        'ein', 'eine', 'einer', 'einem', 'einen', 'eines',
        'dieser', 'diese', 'dieses', 'jene', 'jener', 'jenes',
        'mein', 'meine', 'dein', 'deine', 'sein', 'seine', 'ihr', 'ihre',
        'unser', 'unsere', 'euer', 'eure',
        'mancher', 'mehrere', 'viele', 'alle', 'keiner', 'jede', 'jeder', 'jedes'
    },

    # 🇮🇹 Italian
    'it': {
        'il', 'lo', 'la', 'i', 'gli', 'le', 'uno', 'una', 'un',
        'questo', 'questa', 'questi', 'queste', 'quello', 'quella', 'quei', 'quegli', 'quelle',
        'mio', 'mia', 'miei', 'mie', 'tuo', 'tua', 'tuoi', 'tue', 'suo', 'sua', 'suoi', 'sue',
        'nostro', 'nostra', 'nostri', 'nostre', 'vostro', 'vostra', 'vostri', 'vostre',
        'ogni', 'alcuni', 'alcune', 'tutti', 'tutte', 'nessuno'
    },

    # 🇵🇹 Portuguese
    'pt': {
        'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
        'do', 'da', 'dos', 'das', 'ao', 'à', 'aos', 'às',
        'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas',
        'aquele', 'aquela', 'aqueles', 'aquelas',
        'meu', 'minha', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas',
        'seu', 'sua', 'seus', 'suas', 'nosso', 'nossa', 'nossos', 'nossas',
        'todo', 'toda', 'todos', 'todas', 'algum', 'alguma', 'alguns', 'algumas',
        'nenhum', 'nenhuma', 'cada'
    },

    # 🇷🇺 Russian
    'ru': {
        'этот', 'эта', 'эти', 'тот', 'та', 'те',
        'мой', 'моя', 'мои', 'твой', 'твоя', 'твои', 'его', 'её', 'их',
        'наш', 'наша', 'наши', 'ваш', 'ваша', 'ваши',
        'все', 'каждый', 'некоторый', 'какой-то', 'любой', 'никакой', 'весь'
    },

    # 🇯🇵 Japanese
    'ja': {
        'この', 'その', 'あの', 'どの',
        'ある', 'そのような', 'こうした', 'あらゆる', 'すべての'
    },

    # 🇨🇳 Chinese (Simplified)
    'zh': {
        '这', '那', '这些', '那些',
        '每', '各', '所有', '一些', '任何', '某', '全部',
        '我的', '你的', '他的', '她的', '它的', '我们的', '他们的', '她们的'
    },

    # 🇰🇷 Korean
    'ko': {
        '이', '그', '저', '어느',
        '모든', '각', '어떤', '아무', '몇몇', '모두',
        '내', '너의', '그의', '그녀의', '우리의', '그들의'
    },

    # 🇸🇪 Swedish
    'sv': {
        'en', 'ett', 'den', 'det', 'de',
        'min', 'mitt', 'mina', 'din', 'ditt', 'dina',
        'hans', 'hennes', 'dess', 'vår', 'vårt', 'våra',
        'er', 'ert', 'era',
        'denna', 'detta', 'dessa',
        'varje', 'någon', 'något', 'några', 'alla', 'ingen', 'inget'
    },

    # 🇸🇦 Arabic
    'ar': {
        'ال',  # the definite article prefix “al-”
        'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك',  # demonstratives
        'كل', 'بعض', 'أي', 'أيّ', 'عدة', 'كلّ', 'أيضا', 'أيّما',
        'ما', 'أيها', 'أيهنّ', 'كلا', 'كلتا', 'كلهم',  # quantifiers
        'لي', 'لك', 'له', 'لها', 'لهم', 'لنا', 'لكم', 'لكن', 'لكنها'  # possessives (constructed forms)
    },

    # 🇳🇱 Dutch
    'nl': {
        'de', 'het', 'een',
        'dit', 'dat', 'deze', 'die',
        'mijn', 'jouw', 'zijn', 'haar', 'ons', 'onze', 'hun', 'uw',
        'ieder', 'elk', 'alle', 'sommige', 'geen', 'iedereen', 'niemand'
    },

    # 🇹🇷 Turkish
    'tr': {
        'bir',  # a/an
        'bu', 'şu', 'o', 'bunlar', 'şunlar', 'onlar',  # this/that
        'benim', 'senin', 'onun', 'bizim', 'sizin', 'onların',  # possessives
        'her', 'bazı', 'hiçbir', 'tüm', 'bütün', 'bazısı', 'birkaç', 'diğer'  # quantifiers
    }
}

# For global filtering
COMMON_DETERMINERS = set().union(*DETERMINERS.values())


TERMINATORS = {
    # Default (Western languages)
    'other': ['.', '!', '?', '…', '‽', '?!', '!?', '¡', '¿', ';', ':'],

    # Japanese
    'ja': [
        '。',  # full stop
        '！', '？',  # exclamation / question (full-width)
        '．',  # full-width period
        '…', '‥',  # ellipses (both 2- and 3-dot versions)
        '？！', '！？'  # combined forms
    ],

    # Chinese (Simplified + Traditional)
    'zh': [
        '。', '！', '？',  # main enders
        '……', '…',  # ellipses
        '！？”', '？”', '!?', '?!',  # combo punctuation often used in quotes
        '！』', '？』', '！）', '？）'  # frequent quote closers
    ],

    # Korean
    'ko': [
        '.', '!', '?', '。', '！', '？',  # both Latin and CJK
        '…', '‥',
        '?!', '!?'
    ],

    # Arabic
    'ar': [
        '؟',  # Arabic question mark (reversed)
        '!', '‼', '…', '.',  # also used sometimes in MSA
    ],

    # Hindi / Urdu / Sanskrit scripts
    'hi': [
        '।',  # danda (Devanagari full stop)
        '॥',  # double danda (verse or paragraph end)
        '!', '?', '…'
    ],

    # Thai
    'th': [
        '!', '?', '…', '.', 'ฯ', '๚',  # Thai sentence/section marks
    ]
}

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


def _attach_segment_audio_clips_per_segment(
    asr_dump: dict,
    output_dir: str | Path,
    min_duration: float,
    max_duration: float,
) -> dict:
    """
    For each segment in an ASRResponse dump, slice the original audio using [start, end] (seconds).
    - If duration < min_duration, repeat the clip until >= min_duration.
    - If duration > max_duration, trim to max_duration.
    Saves per-segment WAVs and writes their paths to segment['audio_url'].

    Returns the updated dump.
    """
    if not isinstance(asr_dump, dict):
        raise TypeError("asr_dump must be a dict (ASRResponse.model_dump())")

    orig_path = asr_dump.get("audio_url")
    if not orig_path:
        raise ValueError("ASR dump is missing 'audio_url' pointing to the original audio")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original audio
    audio, sr = sf.read(str(orig_path), always_2d=True)  # shape: [T, C]
    n_samples, n_channels = audio.shape
    total_sec = n_samples / float(sr)

    segments = asr_dump.get("segments") or []
    for idx, seg in enumerate(segments):
        start = seg.get("start")
        end = seg.get("end")

        # Validate times
        if start is None or end is None:
            continue
        start = max(0.0, float(start))
        end = max(start, float(end))
        if start >= total_sec:
            continue

        # Clamp end within file
        end = min(end, total_sec)
        if end <= start:
            continue

        s0 = int(round(start * sr))
        s1 = int(round(end * sr))
        clip = audio[s0:s1, :]  # [T, C]
        dur = max(0.0, (s1 - s0) / float(sr))
        if dur == 0.0:
            continue

        # If shorter than min, repeat to exceed or meet min
        if dur < min_duration:
            need = int(math.ceil((min_duration - 1e-9) / dur))
            clip = np.vstack([clip] * max(1, need))

        # If longer than max, trim
        max_len = int(round(max_duration * sr))
        if max_duration > 0 and clip.shape[0] > max_len:
            clip = clip[:max_len, :]

        # Save segment wav
        seg_path = out_dir / f"segment_{idx:04d}_{start:.2f}-{end:.2f}.wav"
        sf.write(str(seg_path), clip, sr)
        seg["audio_url"] = str(seg_path)

    return asr_dump

def _attach_segment_audio_clips_one_per_speaker(
    asr_dump: dict,
    output_dir: str | Path,
    min_duration: float,
    max_duration: float,
) -> dict:
    """
    Build a single reference audio per speaker and assign it to all that speaker's segments.
    - Group segments by `speaker_id`.
    - For each speaker, pick the longest raw segment (by end-start) as the reference.
    - If reference < min_duration, repeat until >= min; if > max_duration (>0), trim to max.
    - Save one WAV per speaker and set seg['audio_url'] to that same file for uniformity.

    Returns the updated dump.
    """
    if not isinstance(asr_dump, dict):
        raise TypeError("asr_dump must be a dict (ASRResponse.model_dump())")

    orig_path = asr_dump.get("audio_url")
    if not orig_path:
        raise ValueError("ASR dump is missing 'audio_url' pointing to the original audio")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original audio (T x C)
    audio, sr = sf.read(str(orig_path), always_2d=True)
    n_samples, _ = audio.shape
    total_sec = n_samples / float(sr)

    segments = asr_dump.get("segments") or []

    # Helper: sanitize speaker id for filenames
    def _safe_name(s):
        s = "unknown" if s in (None, "") else str(s)
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:64]

    # Group segments by speaker_id with their slice metadata
    speakers: dict[str, list[tuple[int, float, float, int, int, float]]] = {}
    for idx, seg in enumerate(segments):
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            continue
        start = max(0.0, float(start))
        end = max(start, float(end))
        if start >= total_sec:
            continue
        end = min(end, total_sec)
        if end <= start:
            continue

        s0 = int(round(start * sr))
        s1 = int(round(end * sr))
        dur = max(0.0, (s1 - s0) / float(sr))
        if dur <= 0.0:
            continue

        spk = seg.get("speaker_id") or "unknown"
        speakers.setdefault(str(spk), []).append((idx, start, end, s0, s1, dur))

    # For each speaker, pick longest raw slice, normalize to [min,max], save once, assign to all segments
    for spk, items in speakers.items():
        if not items:
            continue

        # Choose the longest
        longest = max(items, key=lambda x: x[5])  # x[5] = dur
        _, start, end, s0, s1, dur = longest

        clip = audio[s0:s1, :]  # [T, C]

        # Repeat to reach min_duration (if requested)
        if min_duration and dur < min_duration:
            reps = int(math.ceil((min_duration - 1e-9) / max(dur, 1e-9)))
            clip = np.vstack([clip] * max(1, reps))
            dur = clip.shape[0] / float(sr)

        # Trim to max_duration (if > 0)
        if max_duration and max_duration > 0:
            max_len = int(round(max_duration * sr))
            if clip.shape[0] > max_len:
                clip = clip[:max_len, :]

        # Save per-speaker reference file
        spk_name = _safe_name(spk)
        seg_path = out_dir / f"speaker_{spk_name}_ref.wav"
        sf.write(str(seg_path), clip, sr)

        # Assign same audio_url to all segments of this speaker
        for (idx, *_rest) in items:
            segments[idx]["audio_url"] = str(seg_path)

    # Segments with no valid audio remain unchanged
    return asr_dump

def attach_segment_audio_clips(
    asr_dump: dict,
    output_dir: str | Path,
    min_duration: float,
    max_duration: float,
    one_per_speaker: bool = False,
) -> dict:
    """
    Attach audio clips to each segment in the ASRResponse dump.
    Can either create one clip per segment or one clip per speaker.

    Parameters:
    - asr_dump: dict from ASRResponse.model_dump()
    - output_dir: directory to save segment audio files
    - min_duration: minimum duration (seconds) for each clip
    - max_duration: maximum duration (seconds) for each clip (0 = no limit)
    - one_per_speaker: if True, create one clip per speaker; else one per segment

    Returns the updated dump with 'audio_url' in each segment.
    """
    if one_per_speaker:
        return _attach_segment_audio_clips_one_per_speaker(
            asr_dump, output_dir, min_duration, max_duration
        )
    else:
        return _attach_segment_audio_clips_per_segment(
            asr_dump, output_dir, min_duration, max_duration
        )
#---- Translation Segment Aligner ----

@dataclass
class AlignedSegment:
    original_text: str | None = None
    translated_text: str | None = None
    word_alignments: List[Tuple[int, int]] | None = None
    confidence: float = 0.0
    is_monotonic: bool = True
    source_segment_indices: List[int] = field(default_factory=list)
    target_indices: List[int] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    speaker_id: str | None = None

class Aligner(ABC):
    """Base utilities shared by alignment strategies."""

    sentence_endings = TERMINATORS

    def __init__(self) -> None:
        self.mecab = self._init_mecab()

    def _init_mecab(self):
        try:
            import MeCab
            return MeCab.Tagger()
        except:
            return None

    def _is_cjk(self, char: str) -> bool:
        c = ord(char)
        return (0x4E00 <= c <= 0x9FFF or 0x3040 <= c <= 0x30FF or 
                0xAC00 <= c <= 0xD7AF or 0x3400 <= c <= 0x4DBF)

    def _detect_lang(self, text: str) -> str:
        if not text:
            return "other"
        cjk_ratio = sum(1 for c in text if self._is_cjk(c)) / len(text)
        if cjk_ratio > 0.3:
            if any(0x3040 <= ord(c) <= 0x30FF for c in text):
                return "ja"
            if any(0xAC00 <= ord(c) <= 0xD7AF for c in text):
                return "ko"
            return "zh"
        return "other"

    def _is_punctuation(self, token: str) -> bool:
        return bool(re.match(r'^[^\w\s]+$', token))

    def tokenize(self, text: str) -> List[str]:

        """Fast tokenization with CJK support."""

        lang = self._detect_lang(text)
        if lang not in ["ja", "zh", "ko"]:
            return re.findall(r'\w+|[^\w\s]', text)

        tokens, i, length = [], 0, len(text)
        while i < length:
            c = text[i]
            if c.isalpha() and not self._is_cjk(c):
                j = i
                while i < length and text[i].isalpha()  and not self._is_cjk(text[i]):
                    i += 1
                token = text[j:i]
                if token:
                    tokens.append(token)
            elif c.isdigit():
                j = i
                while i < length and text[i].isdigit():
                    i += 1
                tokens.append(text[j:i])
            elif self._is_cjk(c):
                j = i
                while i < length and self._is_cjk(text[i]):
                    i += 1
                tokens.extend(self._tokenize_cjk(text[j:i], lang))
            elif c.strip() and re.match(r'[^\w\s]', c):
                tokens.append(c)
                i += 1
            else:
                i += 1
        return tokens

    def _tokenize_cjk(self, text: str, lang: str) -> List[str]:
        if lang == "ja" and self.mecab:
            node = self.mecab.parseToNode(text)
            tokens = []
            while node:
                if node.surface:
                    tokens.append(node.surface)
                node = node.next
            return tokens
        if lang == "zh":
            try:
                import jieba
                return [t for t in jieba.cut(text, cut_all=False) if t.strip()]
            except:
                pass
        if lang == "ko":
            try:
                from konlpy.tag import Okt
                return Okt().morphs(text)
            except:
                pass
        return [text]

    def _reconstruct(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        lang = self._detect_lang("".join(tokens[:min(10, len(tokens))]))
        if lang in ["zh", "ja"]:
            result = ""
            for i, token in enumerate(tokens):
                if i > 0:
                    prev = tokens[i - 1]
                    if (token[0].isalnum() and not self._is_cjk(token[0]) and
                            (prev[-1].isalnum() or self._is_cjk(prev[-1]))):
                        result += " "
                    elif (prev[-1].isalnum() and not self._is_cjk(prev[-1]) and
                          self._is_cjk(token[0])):
                        result += " "
                result += token
            return result.strip()
        if lang == "ko":
            return " ".join(tokens)
        result = ""
        for i, token in enumerate(tokens):
            if i > 0 and not (self._is_punctuation(token) or token=="$") and tokens[i - 1] not in {'-', "'", "’", "$"}:
                result += " "
            result += token
        return result.strip()

    def _realign_on_sentence_boundaries_and_determiners(
        self,
        tokens: List[str],
        boundaries: List[Tuple[int, int]],
        lang: str,
        max_look_distance: int = 4,
        verbose: bool = False
    ) -> List[Tuple[int, int]]:
        endings = set(self.sentence_endings.get(lang, self.sentence_endings["other"]))
        b = list(boundaries)
        tok_count = len(tokens)

        if verbose:
            print(f"🔧 Realigning with sentence endings and determiners (look={max_look_distance})")

        for i, (start, end) in enumerate(b):
            if i < len(b) - 1 and end < tok_count:
                q = end
                while q < tok_count and self._is_punctuation(tokens[q]):
                    q += 1
                j = q
                while j < tok_count and j - q <= max_look_distance:
                    if tokens[j] in endings:
                        k = j + 1
                        while k < tok_count and self._is_punctuation(tokens[k]):
                            k += 1
                        end = k
                        if verbose:
                            print(f"   seg {i}: extended to {end} on terminator '{tokens[j]}'")
                        break
                    if self._is_punctuation(tokens[j]):
                        q += 1 # permit to not take punctuation into account in the number of lookahead tokens
                    j += 1
                    

                if end > start and end <= tok_count and tokens[end - 1].lower() in COMMON_DETERMINERS:
                    end -= 1
                    if verbose:
                        print(f"   seg {i}: moved determiner '{tokens[end]}' to seg {i+1}")

                if end < tok_count and end > start and tokens[end - 1] in {"'", "-", "’"}:
                    end = min(end + 1, tok_count)
                    if verbose and end < tok_count:
                        print(f"   seg {i}: moved punctuation '{tokens[end - 1]}' to seg {i+1}")

            if i > 0 and start > 0 and tokens[start - 1] not in endings:
                q = start - 1
                j = q - 1
                while j >= 0 and q - j <= max_look_distance:
                    if tokens[j] in endings:
                        k = j + 1
                        while k < start and self._is_punctuation(tokens[k]):
                            k += 1
                        new_start = min(k, end)
                        if new_start < start:
                            if verbose:
                                print(f"   seg {i}: pulled back to {new_start} on terminator '{tokens[j]}'")
                            start = new_start
                        break
                    if self._is_punctuation(tokens[j]):
                        q -= 1 # permit to not take punctuation into account in the number of lookbehind tokens
                    j -= 1

            b[i] = (start, end)

            if i > 0:
                prev_start, _ = b[i - 1]
                b[i - 1] = (prev_start, max(start, prev_start))
            if i < len(b) - 1:
                _, next_end = b[i + 1]
                b[i + 1] = (end, max(next_end, end))

        return b

    def _assign_timings(
            self,
            aligned: List[AlignedSegment],
            target_tokens: List[str],
            source_metadata: List[dict]
        ) -> None:

        src_meta = {
            idx: (seg["start"], seg["end"])
            for idx, seg in enumerate(source_metadata)
            if "start" in seg and "end" in seg
        }

        print("="*20 + "debugging info" + "="*20)
        print("src_meta : ", src_meta)
        print("="*40)

        total_valid_duration=0
        buckets = []
        for seg in aligned:
            start = min( [src_meta[i][0] for i in seg.source_segment_indices if i in src_meta] )
            end = max( [src_meta[i][1] for i in seg.source_segment_indices if i in src_meta] + [start])
            buckets.append( (start, end) )
            current_duration = end - start
            total_valid_duration += max(current_duration, 0)

        print("="*20 + "debugging info" + "="*20)
        print("buckets : ", buckets)
        print("="*40)

        weights = [max(1, len([1 for tok in seg.target_indices if not self._is_punctuation(target_tokens[tok])])) for seg in aligned]
        total_weight = float(sum(weights)) or 1.0
        current_start = src_meta.get(0)[0]

        print("="*20 + "debugging info" + "="*20)
        print("current_start : ", current_start)
        print("="*40)

        for idx, (seg, weight) in enumerate(zip(aligned, weights)):
            seg_duration = total_valid_duration * weight / total_weight
            seg.start = current_start
            seg.end = current_start + seg_duration
            if idx + 1 in src_meta:
                current_start += seg_duration + src_meta.get(idx + 1)[0] - src_meta.get(idx)[1]
            else:
                seg.end = source_metadata[-1].get("end")

    def _assign_speakers(
            self,
            aligned: List[AlignedSegment],
            source_metadata: List[dict]
        ) -> None:

        src_speakers = {
            idx: seg.get("speaker_id")
            for idx, seg in enumerate(source_metadata)
            if "speaker_id" in seg
        }

        for seg in aligned:
            speakers = [src_speakers[i] for i in seg.source_segment_indices if i in src_speakers]
            if speakers:
                seg.speaker_id = max(set(speakers), key=speakers.count)

    @abstractmethod
    def align_segments(self, source_segments: List[str] | None, translated_text: str,
                      verbose: bool = False, max_look_distance: int = 4, source_metadata: Optional[List[dict]] = None) -> List[AlignedSegment]:
        """Main alignment method."""
        pass


class SophisticatedAligner(Aligner):
    """Fast aligner for dubbing pipelines. Handles CJK and non-monotonic translations."""
    
    def __init__(self, model: str = "bert", token_type: str = "bpe",
                 matching_method: str = "i", allow_merging: bool = False):
        super().__init__()
        self.aligner = SentenceAligner(model=model, token_type=token_type,
                                       matching_methods=matching_method)
        self.matching_method = {"a": "inter", "m": "mwmf", "i": "itermax",
                                "f": "fwd", "r": "rev"}[matching_method]
        self.allow_merging = allow_merging
    

    def _rebuild_segments_from_boundaries(
        self,
        original_segments: List[AlignedSegment],
        source_segments: List[str],
        boundaries: List[Tuple[int, int]],
        tgt_tokens: List[str],
        t2s: Optional[dict] = None
    ) -> List[AlignedSegment]:
        
        def _merge_indices(left: List[int], right: List[int]) -> List[int]:
            seen, merged = set(), []
            for idx in left + right:
                if idx not in seen:
                    seen.add(idx)
                    merged.append(idx)
            return merged

        rebuilt: List[AlignedSegment] = []
        carry_sources: List[int] = []
        carry_aligns: List[Tuple[int, int]] = []

        for i, (start_pos, end_pos) in enumerate(boundaries):
            tgt_idx = list(range(start_pos, end_pos))
            base = original_segments[min(i, len(original_segments) - 1)]

            if not tgt_idx:
                if rebuilt:
                    prev = rebuilt[-1]
                    prev.source_segment_indices = _merge_indices(prev.source_segment_indices, base.source_segment_indices)
                    prev.original_text = " ".join(source_segments[j] for j in prev.source_segment_indices) if prev.source_segment_indices else None
                    if base.word_alignments:
                        prev.word_alignments = (prev.word_alignments or []) + list(base.word_alignments)
                else:
                    carry_sources = _merge_indices(carry_sources, base.source_segment_indices)
                    if base.word_alignments:
                        carry_aligns.extend(base.word_alignments)
                continue

            current_tokens = [tgt_tokens[j] for j in tgt_idx]
            text = self._reconstruct(current_tokens)

            merged_sources = _merge_indices(carry_sources, base.source_segment_indices)
            carry_sources = []
            aligns = [pair for pair in (carry_aligns + (base.word_alignments or []))]
            carry_aligns = []

            if t2s is not None and tgt_idx:
                conf = sum(1 for t in tgt_idx if t in t2s) / len(tgt_idx)
            else:
                conf = base.confidence

            final_sources = merged_sources if merged_sources else base.source_segment_indices
            orig_text = " ".join(source_segments[j] for j in final_sources) if final_sources else None

            rebuilt.append(AlignedSegment(
                original_text=orig_text,
                translated_text=text,
                word_alignments=aligns,
                confidence=conf,
                is_monotonic=base.is_monotonic,
                source_segment_indices=final_sources,
                target_indices=tgt_idx
            ))

        # just happen if every boundiary encountered in the main loop was empty(normally shouldn't)
        # but just in case, we handle it
        if carry_sources and not rebuilt:
            orig_text = " ".join(source_segments[j] for j in carry_sources) if carry_sources else None
            rebuilt.append(AlignedSegment(
                original_text=orig_text,
                translated_text="",
                word_alignments=[],
                confidence=0.0,
                is_monotonic=True,
                source_segment_indices=carry_sources,
                target_indices=[]
            ))

        return rebuilt

    def _enforce_punctuation_and_sentence_rules(
        self,
        segs: List[AlignedSegment],
        source_segments: List[str],
        tgt_tokens: List[str],
        full_target_text: str,
        t2s: Optional[dict],
        verbose: bool = False,
        max_look_distance: int = 4
    ) -> List[AlignedSegment]:
        """
        Apply unified rules to TranslationSegmentAligner output:
        - trailing punctuation sticks to previous segment
        - prefer ending on sentence terminators
        - move dangling determiners at end to next segment
        - prevent empty or punctuation-only segments
        """
        # 1) Construct coarse contiguous boundaries per segment
        #    Use min..max+1 over target_indices; skip empty segments for now

        # Make spans monotonic and non-overlapping
        fixed: List[Tuple[int, int]] = []
        last_end = 0
        for seg in segs:
            s = min(seg.target_indices) if seg.target_indices else last_end
            e = max(seg.target_indices) + 1 if seg.target_indices else last_end

            if e <= s:
                fixed.append((last_end, last_end))
                continue

            s = max(s, last_end)
            e = max(e, s)
            fixed.append((s, e))
            last_end = e

        # 2) Realign on sentence boundaries and determiners
        lang = self._detect_lang(full_target_text)
        realigned = self._realign_on_sentence_boundaries_and_determiners(
            tgt_tokens, fixed, lang, max_look_distance=max_look_distance, verbose=verbose
        )

        # 3) Rebuild segments with new boundaries
        rebuilt = self._rebuild_segments_from_boundaries(segs, source_segments, realigned, tgt_tokens, t2s=t2s)
        return rebuilt
    
    def _filter_punct(self, tokens: List[str]) -> Tuple[List[str], dict]:
        clean, idx_map = [], {}
        for i, t in enumerate(tokens):
            if not re.match(r'^[^\w\s]+$', t):
                idx_map[len(clean)] = i
                clean.append(t)
        return clean, idx_map

    def _create_segments(self, bounds, src_map, tgt_map, s2t, t2s, is_mono):
        segments, used = [], set()
        
        for idx, (start, end) in enumerate(bounds):

            # original = [src_tokens[tok] for tok in src_map.values() if start <= tok < end]
            # Get aligned target indices
            tgt_idx = set()
            for s in range(start, end):
                if s in s2t:
                    tgt_idx.update(s2t[s])
            tgt_idx -= used

            final = tgt_idx
            # Determine final indices
            if not tgt_idx:
                final = self._fallback(idx, bounds, src_map, tgt_map, used)
            elif is_mono:
                final = list(range(min(tgt_idx), max(tgt_idx) + 1))
            
            used.update(final)
            final = sorted(final)
            
            # Build segment
            aligns = [(s, t) for t in final if t in t2s for s in t2s[t] if start <= s < end]
            conf = sum(1 for i in final if i in t2s) / len(final) if final else 0.0
            
            segments.append(AlignedSegment(
                None, None, aligns, conf, is_mono, [idx], final
            ))
        
        return segments
    
    def _merge_matrix(self, segs: List[AlignedSegment]) -> np.ndarray:
        n = len(segs)
        if n <= 1:
            return np.zeros(0, dtype=bool)

        flags = np.zeros(n - 1, dtype=bool)
        last_non_empty = max((i for i, s in enumerate(segs) if s.target_indices), default=-1)

        empty_run_start = None
        for i in range(n - 1):
            cur_idx = segs[i].target_indices

            if not cur_idx:
                empty_run_start = i if empty_run_start is None else empty_run_start
                continue

            if empty_run_start is not None:
                flags[empty_run_start:i] = True
                empty_run_start = None

            if i >= last_non_empty:
                break

            max1 = max(cur_idx)
            for j in range(i + 1, last_non_empty + 1):
                next_idx = segs[j].target_indices
                if not next_idx:
                    continue
                if max1 >= min(next_idx):
                    flags[i] = True
                    break

        if empty_run_start is not None:
            flags[empty_run_start:] = True

        return flags

    def _handle_reorder(self, segs, src_segs, flags) -> List[AlignedSegment]:
        result, i = [], 0
        n = len(segs)

        while i < n:
            j = i
            while j < n - 1 and flags[j]:
                j += 1

            if j > i:
                group = segs[i:j + 1]
                if self.allow_merging:
                    result.append(self._merge(group))
                else:
                    result.extend(self._redistribute(group, src_segs))
                i = j + 1
            else:
                result.append(segs[i])
                i += 1

        return result
    
    def _merge(self, group) -> AlignedSegment:
        src_idx = [i for s in group for i in s.source_segment_indices]
        
        tgt_idx = set()
        for s in group:
            tgt_idx.update(s.target_indices)
        
        
        aligns = [a for s in group for a in s.word_alignments]
        conf = sum(s.confidence for s in group) / len(group)

        original = " ".join([s.original_text for s in group if s.original_text]) or None

        return AlignedSegment(original, None, aligns, conf, True, src_idx, sorted(tgt_idx))

    def _redistribute(self, group, src_segs) -> List[AlignedSegment]:
        non_empty = [seg for seg in group if seg.target_indices]
        if not non_empty:
            return group

        all_tgt = sorted({t for seg in non_empty for t in seg.target_indices})
        lens = [
            sum(len(src_segs[idx]) for idx in seg.source_segment_indices)
            for seg in non_empty
        ]
        total = sum(lens)
        splits = [max(1, int(round(length / total * len(all_tgt)))) for length in lens] if total else [len(all_tgt)]

        diff = len(all_tgt) - sum(splits)
        for offset in range(abs(diff)):
            if diff > 0:
                splits[offset % len(splits)] += 1
            else:
                k = max(range(len(splits)), key=lambda m: splits[m])
                if splits[k] > 1:
                    splits[k] -= 1

        result, pos = [], 0
        for seg, slen in zip(non_empty, splits):
            idx = all_tgt[pos:pos + slen] if all_tgt else []
            result.append(AlignedSegment(seg.original_text, None, seg.word_alignments,
                                         seg.confidence, seg.is_monotonic,
                                         seg.source_segment_indices, idx))
            pos += slen

        return result
    
    def _fill_gaps(self, segs, tgt_toks, used, verbose):
        unused = sorted(set(range(len(tgt_toks))) - used)
        if not unused:
            return segs
        
        if verbose:
            print(f"⚠️  Filling {len(unused)} gaps")
        
        result = []
        for i, seg in enumerate(segs):

            # Skip if empty segment. NB: normally shouldn't happen after redistribution or merging
            if not seg.target_indices:
                result.append(seg)
                continue

            idx = seg.target_indices
            seg_min = 0 if i == 0 else min(idx)
            
            # update range to include any tokens in the range since now all segments are monotonic and non-overlapping
            if i < len(segs) - 1 and segs[i + 1].target_indices:
                next_min = min(segs[i + 1].target_indices)
                new_rng = list(range(seg_min, next_min))
            else:
                new_rng = list(range(seg_min, len(tgt_toks)))
            if new_rng:
                seg = AlignedSegment(seg.original_text, None, seg.word_alignments,
                                    seg.confidence, seg.is_monotonic, 
                                    seg.source_segment_indices, new_rng)

            result.append(seg)
        
        return result

    def _fallback(self, idx, bounds, src_map, tgt_map, used):
        start, end = bounds[idx]
        total_src = len(src_map.values())
        ratio = len([i for i in range(start, end) if i in src_map]) / total_src if total_src > 0 else 0
        unused = [i for i in tgt_map.values() if i not in used]
        return unused[:max(1, int(len(unused) * ratio))]

    def align_segments(self, source_segments: List[str] | None, translated_text: str,
                      verbose: bool = False, max_look_distance: int = 4, source_metadata: Optional[List[dict]] = None) -> List[AlignedSegment]:
        
        """Main alignment method."""
        
        if not source_segments and not source_metadata:
            print ("⚠️  No source segments or metadata provided; Nothing to align.")
            return None
        
        elif not source_segments and source_metadata:
            source_segments = [seg["text"] for seg in source_metadata if "text" in seg]
            print ("👍  Source segments extracted from metadata.")
            print (source_segments)

        if not source_segments:
            print("⚠️  No source text available after metadata extraction.")
            return None


        if verbose:
            print(f"\n{'='*70}\n🔍 Aligning {len(source_segments)} segments\n{'='*70}")
        
        # Tokenize
        src_tokens, boundaries = [], []
        pos = 0
        for seg in source_segments:
            tokens = self.tokenize(seg)
            src_tokens.extend(tokens)
            boundaries.append((pos, pos + len(tokens)))
            pos += len(tokens)
        
        tgt_tokens = self.tokenize(translated_text)
        
        # Filter punctuation
        src_clean, src_map = self._filter_punct(src_tokens)
        tgt_clean, tgt_map = self._filter_punct(tgt_tokens)
        
        if verbose:
            print(f"📝 Tokens: {len(src_tokens)} src → {len(tgt_tokens)} tgt")
        
        # Align
        alignments = self.aligner.get_word_aligns(src_clean, tgt_clean)[self.matching_method]
        align_orig = [(src_map[s], tgt_map[t]) for s, t in alignments]
        is_mono = all(align_orig[i][1] <= align_orig[i+1][1] for i in range(len(align_orig)-1)) if align_orig else True
        
        if verbose:
            print(f"🔗 {len(alignments)} alignments ({'✓ mono' if is_mono else '⚠️  reordered'})")
        
        # Build maps
        s2t, t2s = {}, {}
        for s, t in align_orig:
            s2t.setdefault(s, []).append(t)
            t2s.setdefault(t, []).append(s)
        
        # Create segments
        segments = self._create_segments(boundaries, src_map, tgt_map, s2t, t2s,
                                          is_mono)

        print(f"🧩 Initial segments: {len(segments)}")
        for i, seg in enumerate(segments):
            print(f"   seg {i}: '{seg.target_indices}' from source segs {seg.source_segment_indices}")

        # Handle reordering
        if not is_mono and len(segments) > 1:
            merge_flag = self._merge_matrix(segments)
            if np.any(merge_flag):
                segments = self._handle_reorder(segments, source_segments, merge_flag)

            print(f"🔄 After reordering handling: {len(segments)} segments")
            for i, seg in enumerate(segments):
                print(f"   seg {i}: '{seg.target_indices}' from source segs {seg.source_segment_indices}")
        
        # Complete coverage
        used = set()
        for seg in segments:
            used.update(seg.target_indices)
        segments = self._fill_gaps(segments, tgt_tokens, used, verbose)

        print(f"🔄 After gap filling: {len(segments)} segments")
        for i, seg in enumerate(segments):
            print(f"   seg {i}: '{seg.target_indices}' from source segs {seg.source_segment_indices}")


        #  enforce punctuation/sentence/determiner rules (unified behavior)
        segments = self._enforce_punctuation_and_sentence_rules(
            segs=segments,
            source_segments=source_segments,
            tgt_tokens=tgt_tokens,
            full_target_text=translated_text,
            t2s=t2s,
            verbose=verbose,
            max_look_distance=max_look_distance
        )
        if source_metadata:
            self._assign_timings(segments, tgt_tokens, source_metadata)
            self._assign_speakers(segments, source_metadata)
        if verbose:
            print(f"✅ Done: {len(segments)} segments\n{'='*70}\n")
        return segments


class ProportionalAligner(Aligner):

    """Simple proportional aligner with smart punctuation-aware boundaries."""
    
    def __init__(self):
        super().__init__()
    
    def align_segments(self, source_segments: List[str] | None, translated_text: str,
                      verbose: bool = False, max_look_distance: int = 4, source_metadata: Optional[List[dict]] = None) -> List[AlignedSegment]:
        
        """Proportionally allocate target tokens with smart punctuation handling."""

        if not source_segments and not source_metadata:
            print ("⚠️  No source segments or metadata provided; Nothing to align.")
            return None
        
        elif not source_segments and source_metadata:
            source_segments = [seg["text"] for seg in source_metadata if "text" in seg]

        if not source_segments:
            print("⚠️  No source text available after metadata extraction.")
            return None

        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 Proportional Alignment: {len(source_segments)} segments")
            print(f"{'='*70}")
        
        # Count words per source segment
        source_word_counts = []
        for seg in source_segments:
            tokens = self.tokenize(seg)
            word_count = sum(1 for t in tokens if not self._is_punctuation(t))
            source_word_counts.append(word_count)
        
        total_source_words = sum(source_word_counts)
        
        # Tokenize target
        target_tokens = self.tokenize(translated_text)
        target_word_indices = [i for i, t in enumerate(target_tokens) if not self._is_punctuation(t)]
        total_target_words = len(target_word_indices)
        target_lang = self._detect_lang(translated_text)
        
        if verbose:
            print(f"📝 Source: {total_source_words} words")
            print(f"📝 Target: {total_target_words} words (lang: {target_lang})")
        
        # Proportional allocation
        if total_source_words == 0:
            words_per_segment = [max(1, total_target_words // len(source_segments))] * len(source_segments)
            remainder = total_target_words % len(source_segments)
            for i in range(remainder):
                words_per_segment[i] += 1
        else:
            words_per_segment = [
                max(1, int(round(count / total_source_words * total_target_words)))
                for count in source_word_counts
            ]
        
        # Adjust to match total
        diff = total_target_words - sum(words_per_segment)
        for _ in range(abs(diff)):
            if diff > 0:
                words_per_segment[_ % len(words_per_segment)] += 1
            else:
                max_idx = max((idx for idx, val in enumerate(words_per_segment) if val > 1),
                             key=lambda idx: words_per_segment[idx], default=0)
                words_per_segment[max_idx] -= 1
        
        # Create word-based boundaries
        word_boundaries = []
        word_pos = 0
        for word_count in words_per_segment:
            start_idx = target_word_indices[word_pos] if word_pos < len(target_word_indices) else len(target_tokens)
            end_pos = min(word_pos + word_count, len(target_word_indices))
            end_idx = target_word_indices[end_pos - 1] + 1 if end_pos > 0 else start_idx
            
            # Attach trailing punctuation
            while end_idx < len(target_tokens) and self._is_punctuation(target_tokens[end_idx]):
                end_idx += 1
            
            word_boundaries.append((start_idx, end_idx))
            word_pos = end_pos
        
        # Realign on sentence boundaries
        final_boundaries = self._realign_on_sentence_boundaries_and_determiners(
            target_tokens, word_boundaries, target_lang, max_look_distance, verbose
        )
        
        # Create segments
        aligned_segments = []
        for i, (start_pos, end_pos) in enumerate(final_boundaries):
            if end_pos - start_pos == 0:
                continue

            segment_tokens = target_tokens[start_pos:end_pos]
            translated_seg = self._reconstruct(segment_tokens)

            source_indices = [min(i, len(source_segments) - 1)]
            if i + 1 < len(final_boundaries) and final_boundaries[i+1][1] - final_boundaries[i+1][0] == 0:
                source_indices.append(min(i + 1, len(source_segments) - 1))
            if i == 1 and final_boundaries[0][1] - final_boundaries[0][0] == 0:
                source_indices.insert(0, 0)
                
            original_text = " ".join(source_segments[j] for j in source_indices)
            
            aligned_seg = AlignedSegment(
                original_text=original_text,
                translated_text=translated_seg,
                word_alignments=[],
                confidence=1.0,
                is_monotonic=True,
                source_segment_indices=source_indices,
                target_indices=list(range(start_pos, end_pos))
            )
            
            aligned_segments.append(aligned_seg)
        
        # Assign timestamps
        if source_metadata:
            self._assign_timings(aligned_segments, target_tokens, source_metadata)
            self._assign_speakers(aligned_segments, source_metadata)
        
        if verbose:
            print(f"✅ Done: {len(aligned_segments)} segments")
            for i, seg in enumerate(aligned_segments):
                ts = f" [{seg.start:.2f}s - {seg.end:.2f}s]" if seg.start is not None else ""
                print(f"[{i}] {seg.original_text[:30]:30s} → {seg.translated_text[:40]:40s}{ts}")
            print(f"{'='*70}\n")
        
        return aligned_segments


def alignerWrapper(input_dict, translation_strategy, target_lang, max_look_distance=4, verbose=True):

    aligned_translations = []
    
    for i in input_dict.keys():
        
        # Align translation back to original segments
        if translation_strategy in ["long_proportional", "long_default"]:
            aligner = ProportionalAligner()
            aligned_translation = aligner.align_segments(
                source_segments=None,
                translated_text=input_dict[i]["full_text"],
                verbose=verbose,
                max_look_distance=max_look_distance,
                source_metadata=input_dict[i]["segments"]
                
            )
        elif translation_strategy in ["long_sophisticated", "long_sophisticated_merging"]:
            allow_merging = translation_strategy == "long_sophisticated_merging"
            aligner = SophisticatedAligner(matching_method="i", allow_merging=allow_merging)
            aligned_translation = aligner.align_segments(
                source_segments=None,
                translated_text=input_dict[i]["full_text"],
                verbose=verbose,
                max_look_distance=max_look_distance,
                source_metadata=input_dict[i]["segments"]
            )

        else:
            # Fallback to proportional aligner
            aligner = ProportionalAligner()
            aligned_translation = aligner.align_segments(
                source_segments=None, 
                translated_text=input_dict[i]["full_text"],
                verbose=verbose,
                max_look_distance=max_look_distance,
                source_metadata=input_dict[i]["segments"]
            )

        aligned_translations.extend( aligned_translation )

    # Each segment ready for TTS synthesis
    Tresponse_segments = ASRResponse()

    # Update translation segments with aligned text
    for i, seg in enumerate(aligned_translations):

        print("="*40)
        print("="*20 + " Debug: Sophisticated Aligner Output " + "*"*20)
        print("="*40)

        print(f"Segment {i}: '{seg.original_text}' → '{seg.translated_text}'")
        T_segment = Segment(
            start=seg.start,
            end=seg.end,
            text=seg.translated_text,
            lang=target_lang,
            speaker_id=seg.speaker_id
        )
        Tresponse_segments.segments.append(T_segment)

    Tresponse_segments.language = target_lang
    tr_result = Tresponse_segments  # Replace with aligned segments

    return tr_result

    
import re

def map_by_text_overlap(coarse, fine):
    def normalize(text: str) -> str:
        return re.sub(r"\W+", " ", (text or "").lower()).strip()

    # Precompute coarse caches
    coarse_cache = []
    for idx, seg in enumerate(coarse):
        raw = normalize(seg.get("text", ""))
        words = raw.split()
        coarse_cache.append({
            "idx": idx,
            "words": set(words),
            "word_count": max(1, len(words)),
            "chars": len(raw.replace(" ", "")),
        })

    mappings = []
    for fi, fseg in enumerate(fine):
        raw_f = normalize(fseg.get("text", ""))
        words_f = raw_f.split()
        if not words_f:
            mappings.append({"fine_idx": fi, "parent_coarse_idx": None, "similarity": 0.0})
            continue

        word_set_f = set(words_f)
        word_count_f = max(1, len(words_f))
        char_count_f = len(raw_f.replace(" ", ""))

        best_ci, best_score = None, 0.0
        for cc in coarse_cache:
            overlap_words = len(word_set_f & cc["words"])
            if overlap_words == 0:
                continue

            containment = overlap_words / word_count_f
            char_overlap = min(char_count_f, cc["chars"]) / max(char_count_f, cc["chars"] or 1)
            score = 0.7 * containment + 0.3 * char_overlap

            if score > best_score:
                best_ci, best_score = cc["idx"], score

        mappings.append({
            "fine_idx": fi,
            "parent_coarse_idx": best_ci,
            "similarity": round(best_score, 3),
        })

        grouped = {
            ci: {
                "segments": [fine[m["fine_idx"]] for m in mappings if m["parent_coarse_idx"] == ci],
                "full_text": coarse[ci].get("text", "")
            }
            for ci in range(len(coarse))
        }

    return grouped