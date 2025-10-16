from .models import Word, Segment
from typing import List, Tuple, Optional, Set
from simalign import SentenceAligner
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Set 
import numpy as np


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
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'de la', 'de l\'', 'l\'',
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
        'il', 'lo', 'la', 'l\'', 'i', 'gli', 'le', 'uno', 'una', 'un\'', 'un',
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
@dataclass
class AlignedSegment:
    original_text: str
    translated_text: str
    word_alignments: List[Tuple[int, int]]
    confidence: float = 0.0
    is_monotonic: bool = True
    source_segment_indices: List[int] = field(default_factory=list)
    target_indices: List[int] = field(default_factory=list)


class TranslationSegmentAligner:
    """Fast aligner for dubbing pipelines. Handles CJK and non-monotonic translations."""
    
    def __init__(self, model: str = "bert", token_type: str = "bpe", 
                 matching_method: str = "i", allow_merging: bool = False):
        self.aligner = SentenceAligner(model=model, token_type=token_type, 
                                       matching_methods=matching_method)
        self.matching_method = {"a": "inter", "m": "mwmf", "i": "itermax", 
                               "f": "fwd", "r": "rev"}[matching_method]
        self.allow_merging = allow_merging
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
    
    def tokenize(self, text: str) -> List[str]:
        """Fast tokenization with CJK support and contraction handling."""
        lang = self._detect_lang(text)
        
        # Non-CJK: word tokenization with contraction handling
        if lang not in ["ja", "zh", "ko"]:
            # Modified regex to keep contractions together (e.g., j'ai, I'm, don't)
            return re.findall(r'\w+|[^\w\s]', text)
        
        # CJK tokenization with contraction handling
        tokens, i = [], 0
        while i < len(text):
            c = text[i]
            # Latin word (including contractions)
            if c.isalpha() and not self._is_cjk(c):
                j = i
                while i < len(text) and (text[i].isalpha() or text[i] in ["'", "'"]) and not self._is_cjk(text[i]):
                    i += 1
                token = text[j:i]
                # Clean up trailing apostrophes
                token = token.rstrip("''")
                if token:
                    tokens.append(token)
            # Digit
            elif c.isdigit():
                j = i
                while i < len(text) and text[i].isdigit():
                    i += 1
                tokens.append(text[j:i])
            # CJK
            elif self._is_cjk(c):
                j = i
                while i < len(text) and self._is_cjk(text[i]):
                    i += 1
                tokens.extend(self._tokenize_cjk(text[j:i], lang))
            # Punctuation (but not apostrophes within words)
            elif c.strip() and re.match(r'[^\w\s]', c) and c not in ["'", "'"]:
                tokens.append(c)
                i += 1
            # Skip whitespace and standalone apostrophes
            else:
                i += 1
        return tokens


    def _reconstruct(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens, merging contractions and spacing correctly."""
        if not tokens:
            return ""

        # Detect language
        text_sample = "".join(tokens[:10])
        lang = self._detect_lang(text_sample)

        # Chinese/Japanese: minimal spacing
        if lang in ["zh", "ja"]:
            result = ""
            for i, token in enumerate(tokens):
                if i > 0:
                    prev_token = tokens[i - 1]
                    if (token[0].isalnum() and not self._is_cjk(token[0]) and
                        (prev_token[-1].isalnum() or self._is_cjk(prev_token[-1]))):
                        result += " "
                    elif (prev_token[-1].isalnum() and not self._is_cjk(prev_token[-1]) and
                          self._is_cjk(token[0])):
                        result += " "
                result += token
            return result.strip()

        # Korean
        if lang == "ko":
            return " ".join(tokens)

        # Other languages: merge contractions, then space before non-punctuation
        result = ""
        for i, token in enumerate(tokens):
            if i > 0 and not self._is_punctuation(token) and tokens[i - 1] != '-' and tokens[i - 1] != "'":
                result += " "
            result += token
        return result.strip()

    def _is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation."""
        return bool(re.match(r'^[^\w\s]+$', token))
    
    def _attach_trailing_punctuation(
        self, tokens: List[str], boundaries: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Ensure punctuation right after a segment end belongs to that segment.
        Boundaries are non-overlapping [start, end) token spans.
        """
        adjusted = []
        for i, (start, end) in enumerate(boundaries):
            next_start = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(tokens)
            j = end
            while j < next_start and j < len(tokens) and self._is_punctuation(tokens[j]):
                j += 1
            adjusted.append((start, j))
        return adjusted

    def _last_non_punct_index(self, tokens: List[str], start: int, end: int) -> int | None:
        """Return index of last non-punctuation token in [start, end), or None."""
        i = end - 1
        while i >= start:
            if not self._is_punctuation(tokens[i]):
                return i
            i -= 1
        return None

    def _is_determiner_at(self, tokens: List[str], idx: int) -> Tuple[bool, int]:
        """
        Detect a determiner starting at idx. Supports:
        - single-token determiners (e.g., 'the', 'le', 'de')
        - simple split forms like ["l", "’"] / ["l", "'"] treated as 2-token determiner
        Returns (is_determiner, span_len).
        """
        if idx < 0 or idx >= len(tokens):
            return False, 0

        t = tokens[idx].lower()

        # single-token determiner
        if t in COMMON_DETERMINERS:
            return True, 1

        # word + apostrophe (idx is word)
        if idx + 1 < len(tokens) and tokens[idx + 1] in ("'", "’"):
            combo = (t + tokens[idx + 1]).lower()
            if combo in COMMON_DETERMINERS:
                return True, 2

        # apostrophe at idx (rare end-at-idx case): prev + apostrophe
        if tokens[idx] in ("'", "’") and idx - 1 >= 0:
            combo = (tokens[idx - 1].lower() + tokens[idx]).lower()
            if combo in COMMON_DETERMINERS:
                return True, 2

        return False, 0

    def _realign_on_sentence_boundaries_and_determiners(
        self,
        tokens: List[str],
        boundaries: List[Tuple[int, int]],
        lang: str,
        max_look_distance: int = 3,
        verbose: bool = False
    ) -> List[Tuple[int, int]]:
        """
        - Prefer ending segments at sentence terminators (TERMINATORS[lang])
        - Keep trailing punctuation with previous segment
        - Move dangling determiners at end of a segment to the next segment
        """
        endings = TERMINATORS.get(lang, TERMINATORS['other'])
        b = list(boundaries)  # copy

        if verbose:
            print(f"🔧 Realigning with sentence endings and determiners (look={max_look_distance})")

        for i, (start, end) in enumerate(b):
            # Extend end forward to include nearby sentence ending
            if i < len(b) - 1:
                next_start = b[i + 1][0]
                j = end
                while j < len(tokens) and j < next_start and j - end <= max_look_distance:
                    if tokens[j] in endings:
                        # include terminator and any following punctuation
                        k = j + 1
                        while k < next_start and k < len(tokens) and self._is_punctuation(tokens[k]):
                            k += 1
                        end = k
                        if verbose:
                            print(f"   seg {i}: extended to {end} on terminator '{tokens[j]}'")
                        break
                    j += 1

            # If last non-punct token at end is a determiner, move it to next segment
            if i < len(b) - 1:
                last_np = self._last_non_punct_index(tokens, start, end)
                if last_np is not None:
                    is_det, span = self._is_determiner_at(tokens, last_np)
                    if is_det:
                        # Move the determiner span [last_np, last_np+span) into the next segment
                        det_start = last_np
                        det_end_excl = min(end, last_np + span)

                        new_end = max(start, det_start)  # cut current before the determiner
                        if new_end < end:
                            if verbose:
                                det_text = "".join(tokens[det_start:det_end_excl])
                                print(f"   seg {i}: moved determiner '{det_text}' to seg {i+1}")
                            # shrink current
                            end = new_end
                            # grow next (move its start back to include the determiner)
                            ns, ne = b[i + 1]
                            b[i + 1] = (min(det_start, ns), max(ne, det_start))

            b[i] = (start, end)

        return b

    def _compact_and_prevent_empty(
        self,
        tokens: List[str],
        boundaries: List[Tuple[int, int]],
        source_segments: List[AlignedSegment],
        verbose: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Prevent empty or punctuation-only target slices.
        Strategy:
        - Drop zero-length spans
        - Merge punctuation-only spans into previous if possible, else next
        - Ensure boundaries are still non-overlapping and ordered
        """
        out: List[Tuple[int, int]] = []
        for i, (s, e) in enumerate(boundaries):
            if e <= s:
                # skip empty; it will be absorbed by neighbor implicitly
                if verbose:
                    print(f"   seg {i}: dropped empty slice")
                continue
            slice_tokens = tokens[s:e]
            non_punct = [t for t in slice_tokens if not self._is_punctuation(t)]
            if not non_punct:
                # punctuation only: attach to previous if any
                if out:
                    ps, pe = out[-1]
                    out[-1] = (ps, e)  # extend previous to include this punct-only area
                    if verbose:
                        print(f"   seg {i}: merged punct-only into previous")
                else:
                    # no previous: push into next by creating a placeholder that next will expand from
                    # Effectively, we skip here; next segment will start earlier by virtue of start continuity
                    if verbose:
                        print(f"   seg {i}: leading punct-only dropped (will be covered by next)")
                continue
            out.append((s, e))

        # Ensure monotonic non-overlapping, fix accidental overlaps
        fixed: List[Tuple[int, int]] = []
        last_end = 0
        for i, (s, e) in enumerate(out):
            s = max(s, last_end)
            e = max(e, s)
            fixed.append((s, e))
            last_end = e
        return fixed

    def _rebuild_segments_from_boundaries(
        self,
        original_segments: List[AlignedSegment],
        boundaries: List[Tuple[int, int]],
        tgt_tokens: List[str],
        t2s: Optional[dict] = None
    ) -> List[AlignedSegment]:
        """
        Rebuild AlignedSegment list from new [start, end) boundaries.
        Keeps source_segment_indices; trims word_alignments to new target range.
        """
        rebuilt: List[AlignedSegment] = []
        for i, (start_pos, end_pos) in enumerate(boundaries):
            tgt_idx = list(range(start_pos, end_pos))
            text = self._reconstruct([tgt_tokens[j] for j in tgt_idx])

            # Base from original segment i if exists, else last
            base = original_segments[min(i, len(original_segments) - 1)]

            # Trim alignments to the new range
            if base.word_alignments:
                aligns = [(s, t) for (s, t) in base.word_alignments if start_pos <= t < end_pos]
            else:
                aligns = []

            # Confidence: ratio of target tokens that have alignment (if t2s provided)
            if t2s is not None and tgt_idx:
                conf = sum(1 for t in tgt_idx if t in t2s) / max(1, len(tgt_idx))
            else:
                conf = base.confidence

            rebuilt.append(AlignedSegment(
                original_text=base.original_text,
                translated_text=text,
                word_alignments=aligns,
                confidence=conf,
                is_monotonic=base.is_monotonic,
                source_segment_indices=base.source_segment_indices,
                target_indices=tgt_idx
            ))
        return rebuilt

    def _enforce_punctuation_and_sentence_rules(
        self,
        segs: List[AlignedSegment],
        tgt_tokens: List[str],
        full_target_text: str,
        t2s: Optional[dict],
        verbose: bool = False,
        max_look_distance: int = 3
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
        coarse: List[Tuple[int, int]] = []
        for s in segs:
            if s.target_indices:
                start = min(s.target_indices)
                end = max(s.target_indices) + 1
                coarse.append((start, end))
            else:
                # placeholder empty span; will be removed/absorbed later
                coarse.append((0, 0))

        # Make spans monotonic and non-overlapping
        fixed: List[Tuple[int, int]] = []
        last_end = 0
        for (s, e) in coarse:
            if e <= s:
                fixed.append((last_end, last_end))
                continue
            s = max(s, last_end)
            e = max(e, s)
            fixed.append((s, e))
            last_end = e

        # 2) Attach trailing punctuation to previous segment
        with_punct = self._attach_trailing_punctuation(tgt_tokens, fixed)

        # 3) Realign on sentence boundaries and determiners
        lang = self._detect_lang(full_target_text)
        realigned = self._realign_on_sentence_boundaries_and_determiners(
            tgt_tokens, with_punct, lang, max_look_distance=max_look_distance, verbose=verbose
        )

        # 4) Prevent empty / punct-only segments
        compact = self._compact_and_prevent_empty(tgt_tokens, realigned, segs, verbose=verbose)

        # 5) Rebuild segments with new boundaries
        rebuilt = self._rebuild_segments_from_boundaries(segs, compact, tgt_tokens, t2s=t2s)
        return rebuilt
    
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
    
    def align_segments(self, source_segments: List[str], translated_text: str, 
                      verbose: bool = False) -> List[AlignedSegment]:
        """Main alignment method."""
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
        segments = self._create_segments(boundaries, source_segments, s2t, t2s, 
                                        tgt_tokens, is_mono, verbose)
        
        # Handle reordering
        if not is_mono and len(segments) > 1:
            merge_mat = self._merge_matrix(segments)
            if np.any(merge_mat):
                segments = self._handle_reorder(segments, source_segments, 
                                               tgt_tokens, merge_mat, verbose)
        
        # Complete coverage
        used = set()
        for seg in segments:
            used.update(seg.target_indices)
        segments = self._fill_gaps(segments, tgt_tokens, used, verbose)
        segments = self._dedup(segments, tgt_tokens)

        # NEW: enforce punctuation/sentence/determiner rules (unified behavior)
        segments = self._enforce_punctuation_and_sentence_rules(
            segs=segments,
            tgt_tokens=tgt_tokens,
            full_target_text=translated_text,
            t2s=t2s,
            verbose=verbose
        )
        
        if verbose:
            print(f"✅ Done: {len(segments)} segments\n{'='*70}\n")
        
        return segments
    
    def _filter_punct(self, tokens: List[str]) -> Tuple[List[str], dict]:
        clean, idx_map = [], {}
        for i, t in enumerate(tokens):
            if not re.match(r'^[^\w\s]+$', t):
                idx_map[len(clean)] = i
                clean.append(t)
        return clean, idx_map
    
    def _create_segments(self, bounds, src_segs, s2t, t2s, tgt_toks, is_mono, verbose):
        segments, used = [], set()
        
        for idx, (start, end) in enumerate(bounds):
            # Get aligned target indices
            tgt_idx = set()
            for s in range(start, end):
                if s in s2t:
                    tgt_idx.update(s2t[s])
            tgt_idx -= used
            
            # Determine final indices
            if not tgt_idx:
                final = self._fallback(idx, bounds, len(tgt_toks), used)
            elif is_mono:
                final = list(range(min(tgt_idx), max(tgt_idx) + 1))
            else:
                final = sorted(tgt_idx)
                # Add gaps
                result = []
                for i, t in enumerate(final):
                    result.append(t)
                    if i < len(final) - 1:
                        for gap in range(t + 1, final[i + 1]):
                            if gap not in t2s and gap not in used:
                                result.append(gap)
                final = sorted(set(result))
            
            used.update(final)
            final = sorted(final)
            
            # Build segment
            text = self._reconstruct([tgt_toks[i] for i in final])
            aligns = [(s, t) for t in final if t in t2s for s in t2s[t]]
            conf = sum(1 for i in final if i in t2s) / len(final) if final else 0.0
            
            segments.append(AlignedSegment(
                src_segs[idx], text, aligns, conf, is_mono, [idx], final
            ))
        
        return segments
    
    def _merge_matrix(self, segs: List[AlignedSegment]) -> np.ndarray:
        n = len(segs)
        mat = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            idx1 = [t for _, t in segs[i].word_alignments]
            if not idx1:
                continue
            min1, max1 = min(idx1), max(idx1)
            
            for j in range(i + 1, n):
                idx2 = [t for _, t in segs[j].word_alignments]
                if not idx2:
                    continue
                min2, max2 = min(idx2), max(idx2)
                
                if (max1 >= min2 and max2 >= min1) or min1 > min2:
                    mat[i, j] = True
        
        return mat
    
    def _handle_reorder(self, segs, src_segs, tgt_toks, mat, verbose):
        result, i = [], 0
        n = len(segs)
        
        while i < n:
            merge_idx = [j for j in range(i + 1, n) if mat[i, j]]
            
            if merge_idx:
                last = max(merge_idx)
                group = segs[i:last + 1]
                
                if self.allow_merging:
                    result.append(self._merge(group, src_segs, tgt_toks))
                else:
                    result.extend(self._redistribute(group, src_segs, tgt_toks))
                
                i = last + 1
            else:
                result.append(segs[i])
                i += 1
        
        return result
    
    def _merge(self, group, src_segs, tgt_toks):
        src_idx = [i for s in group for i in s.source_segment_indices]
        orig = " ".join(src_segs[i] for i in src_idx)
        
        tgt_idx = set()
        for s in group:
            tgt_idx.update([t for _, t in s.word_alignments])
        
        if tgt_idx:
            rng = list(range(min(tgt_idx), max(tgt_idx) + 1))
        else:
            rng = []
        
        text = self._reconstruct([tgt_toks[i] for i in rng])
        aligns = [a for s in group for a in s.word_alignments]
        conf = sum(s.confidence for s in group) / len(group)
        
        return AlignedSegment(orig, text, aligns, conf, True, src_idx, rng)
    
    def _redistribute(self, group, src_segs, tgt_toks):
        all_tgt = []
        for s in group:
            all_tgt.extend([t for _, t in s.word_alignments])
        
        rng = list(range(min(all_tgt), max(all_tgt) + 1)) if all_tgt else []
        
        # Proportional split
        lens = [len(src_segs[i]) for s in group for i in s.source_segment_indices]
        total = sum(lens)
        splits = [max(1, int(round(l / total * len(rng)))) for l in lens] if total > 0 else [1] * len(group)
        
        # Adjust
        diff = len(rng) - sum(splits)
        for j in range(abs(diff)):
            if diff > 0:
                splits[j % len(splits)] += 1
            elif splits[j % len(splits)] > 1:
                splits[j % len(splits)] -= 1
        
        result, pos = [], 0
        for s, slen in zip(group, splits):
            idx = rng[pos:pos + slen] if rng else []
            text = self._reconstruct([tgt_toks[i] for i in idx])
            aligns = [(sr, tg) for sr, tg in s.word_alignments if tg in idx]
            result.append(AlignedSegment(s.original_text, text, aligns, s.confidence,
                                        s.is_monotonic, s.source_segment_indices, idx))
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
            if not seg.word_alignments:
                result.append(seg)
                continue
            
            idx = [t for _, t in seg.word_alignments]
            seg_min, seg_max = min(idx), max(idx)
            
            # Find tokens to add
            if i < len(segs) - 1 and segs[i + 1].word_alignments:
                next_min = min(t for _, t in segs[i + 1].word_alignments)
                to_add = [x for x in unused if seg_max < x < next_min]
            else:
                to_add = [x for x in unused if x > seg_max]
            
            if to_add:
                new_rng = list(range(min(seg_min, min(to_add)), max(seg_max, max(to_add)) + 1))
                text = self._reconstruct([tgt_toks[j] for j in new_rng])
                seg = AlignedSegment(seg.original_text, text, seg.word_alignments,
                                    seg.confidence, seg.is_monotonic, 
                                    seg.source_segment_indices, new_rng)
                unused = [x for x in unused if x not in to_add]
            
            result.append(seg)
        
        return result
    
    def _dedup(self, segs, tgt_toks):
        claimed = set()
        for seg in segs:
            unique = [i for i in seg.target_indices if i not in claimed]
            claimed.update(unique)
            seg.target_indices = unique
            seg.translated_text = self._reconstruct([tgt_toks[i] for i in unique])
        return segs
    
    def _fallback(self, idx, bounds, total_tgt, used):
        start, end = bounds[idx]
        total_src = bounds[-1][1]
        ratio = (end - start) / total_src if total_src > 0 else 0
        unused = [i for i in range(total_tgt) if i not in used]
        return unused[:max(1, int(len(unused) * ratio))]
    
    
    def get_translated_segments(self, source_segments: List[str], 
                               translated_text: str, verbose: bool = False) -> List[str]:
        """Quick export for TTS pipeline."""
        return [s.translated_text for s in self.align_segments(source_segments, translated_text, verbose)]

class ProportionalAligner:
    """Simple proportional aligner with smart punctuation-aware boundaries."""
    
    def __init__(self):
        self.mecab = self._init_mecab()
        # Sentence-ending punctuation marks for different languages (NOT commas!)
        self.sentence_endings = TERMINATORS
    
    def _init_mecab(self):
        try:
            import MeCab
            return MeCab.Tagger()
        except:
            return None
    
    def _is_cjk(self, char: str) -> bool:
        """Check if character is CJK (Chinese, Japanese, Korean)."""
        c = ord(char)
        return (0x4E00 <= c <= 0x9FFF or  # Chinese
                0x3040 <= c <= 0x30FF or  # Japanese Hiragana/Katakana
                0xAC00 <= c <= 0xD7AF or  # Korean Hangul
                0x3400 <= c <= 0x4DBF)    # Chinese Extension A
    
    def _detect_lang(self, text: str) -> str:
        """Detect if text is CJK or other."""
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
        """Check if token is punctuation."""
        return bool(re.match(r'^[^\w\s]+$', token))
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with CJK support and contraction handling."""
        lang = self._detect_lang(text)
        
        # Non-CJK: word tokenization with contraction handling
        if lang == "other":
            # Modified regex to keep contractions together (e.g., j'ai, I'm, don't)
            # Pattern: word characters, optionally followed by apostrophe and more word characters
            return re.findall(r"\w+(?:['\']\w+)?|[^\w\s]", text)
        
        # Japanese: use MeCab if available
        if lang == "ja" and self.mecab:
            node = self.mecab.parseToNode(text)
            tokens = []
            while node:
                if node.surface:
                    tokens.append(node.surface)
                node = node.next
            return tokens
        
        # Chinese: use jieba if available
        if lang == "zh":
            try:
                import jieba
                return [t for t in jieba.cut(text, cut_all=False) if t.strip()]
            except:
                pass
        
        # Korean: use konlpy if available
        if lang == "ko":
            try:
                from konlpy.tag import Okt
                return Okt().morphs(text)
            except:
                pass
        
        # Fallback: character-level tokenization for CJK with contraction handling
        tokens = []
        i = 0
        while i < len(text):
            c = text[i]
            # Keep Latin words together (including contractions)
            if c.isalpha() and not self._is_cjk(c):
                j = i
                while i < len(text) and (text[i].isalpha() or text[i] in ["'", "'"]) and not self._is_cjk(text[i]):
                    i += 1
                token = text[j:i]
                # Clean up trailing apostrophes
                token = token.rstrip("''")
                if token:
                    tokens.append(token)
            # Keep numbers together
            elif c.isdigit():
                j = i
                while i < len(text) and text[i].isdigit():
                    i += 1
                tokens.append(text[j:i])
            # CJK: one character per token
            elif self._is_cjk(c):
                tokens.append(c)
                i += 1
            # Punctuation (but not apostrophes within words - already handled above)
            elif c.strip() and re.match(r'[^\w\s]', c) and c not in ["'", "'"]:
                tokens.append(c)
                i += 1
            # Skip whitespace and standalone apostrophes
            else:
                i += 1
        
        return tokens

    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens, handling spacing and contractions intelligently."""
        if not tokens:
            return ""
        
        # Detect language
        text_sample = "".join(tokens[:10])
        lang = self._detect_lang(text_sample)
        
        # Chinese/Japanese: minimal spacing
        if lang in ["zh", "ja"]:
            result = ""
            for i, token in enumerate(tokens):
                if i > 0:
                    prev_token = tokens[i - 1]
                    # Add space before Latin words/numbers
                    if (token[0].isalnum() and not self._is_cjk(token[0]) and
                        (prev_token[-1].isalnum() or self._is_cjk(prev_token[-1]))):
                        result += " "
                    # Add space after Latin words/numbers before CJK
                    elif (prev_token[-1].isalnum() and not self._is_cjk(prev_token[-1]) and
                        self._is_cjk(token[0])):
                        result += " "
                result += token
            return result.strip()
        
        # Korean: space-separated
        if lang == "ko":
            return " ".join(tokens)
        
        # Other languages: space-separated except before punctuation
        # Contractions are already kept together during tokenization
        result = ""
        for i, token in enumerate(tokens):
            if i > 0:
                # No space before punctuation (but contractions already include apostrophes)
                if not self._is_punctuation(token) and tokens[i - 1] != '-':
                    result += " "
            result += token
        
        return result.strip()
    
    def _separate_words_and_punctuation(self, tokens: List[str]) -> Tuple[List[int], List[int]]:
        """
        Separate token indices into words and punctuation.
        
        Returns:
            Tuple of (word_indices, punct_indices)
        """
        word_indices = []
        punct_indices = []
        
        for i, token in enumerate(tokens):
            if self._is_punctuation(token):
                punct_indices.append(i)
            else:
                word_indices.append(i)
        
        return word_indices, punct_indices
    
    def _attach_punctuation_to_words(self, tokens: List[str], word_boundaries: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Attach punctuation to the previous word segment.
        
        Args:
            tokens: All tokens
            word_boundaries: List of (start, end) boundaries based on words only
        
        Returns:
            Adjusted boundaries including punctuation
        """
        adjusted_boundaries = []
        
        for i, (start, end) in enumerate(word_boundaries):
            adjusted_end = end
            
            # Look ahead for punctuation tokens
            while adjusted_end < len(tokens) and self._is_punctuation(tokens[adjusted_end]):
                adjusted_end += 1
            
            # Make sure we don't overlap with next segment's start
            if i < len(word_boundaries) - 1:
                next_start = word_boundaries[i + 1][0]
                adjusted_end = min(adjusted_end, next_start)
            
            adjusted_boundaries.append((start, adjusted_end))
        
        return adjusted_boundaries
    
    def _find_sentence_endings_in_segments(self, tokens: List[str], boundaries: List[Tuple[int, int]], lang: str) -> List[int]:
        """
        Find segments that end with sentence-ending punctuation.
        
        Returns:
            List of segment indices that end with sentence punctuation
        """
        endings = self.sentence_endings.get(lang, self.sentence_endings['other'])
        sentence_ending_segments = []
        
        for i, (start, end) in enumerate(boundaries):
            if end > start:
                # Check if last token is sentence-ending punctuation
                last_token = tokens[end - 1]
                if last_token in endings:
                    sentence_ending_segments.append(i)
        
        return sentence_ending_segments
    
    def _realign_on_sentence_boundaries(self, tokens: List[str], boundaries: List[Tuple[int, int]], lang: str, verbose: bool = False, max_look_distance = 3) -> List[Tuple[int, int]]:
        """
        Realign segment boundaries to match sentence endings without merging segments.
        
        Strategy:
        - If a segment ends close to a sentence ending (.), move boundary to after the .
        - If a segment starts close to a sentence ending, move that . to previous segment
        - This keeps the same number of segments but makes them more meaningful
        
        Args:
            tokens: All target tokens
            boundaries: List of (start, end) boundaries with punctuation attached
            source_segments: Original source segments
            lang: Language code
            verbose: Debug output
        
        Returns:
            Adjusted boundaries (same count as input)
        """

        # Common determiners in multiple languages
    
        endings = self.sentence_endings.get(lang, self.sentence_endings['other'])
        
        if verbose:
            print(f"🔧 Realigning boundaries on sentence endings (max distance: {max_look_distance} tokens)")
        
        adjusted_boundaries = []
        
        for i, (start, end) in enumerate(boundaries):
            adjusted_start = start
            adjusted_end = end
            
            # Look at the END of current segment - can we extend to a nearby sentence ending?
            if i < len(boundaries) - 1:  # Not the last segment
                found_ending = False
                # Search forward for sentence ending
                for look_pos in range(end, min(end + max_look_distance + 1, len(tokens))):

                    if tokens[look_pos] in endings:
                        while look_pos + 1 < len(tokens) and self._is_punctuation(tokens[look_pos + 1]):
                            look_pos += 1  # Include trailing punctuation
                        # Found a sentence ending nearby!
                        adjusted_end = look_pos + 1  # Include the punctuation
                        found_ending = True
                        if i + 1 < len(boundaries):
                            next_start, next_end = boundaries[i + 1]
                            boundaries[i + 1] = (adjusted_end, max(next_end, adjusted_end))  # Move next segment's start forward

                        if verbose:
                            print(f"   Segment {i}: Extended end from {end} to {adjusted_end} (found '{tokens[look_pos]}')")
                        break

                if not found_ending and tokens[adjusted_end-1] in COMMON_DETERMINERS:
                    # if next token is a determiner (e.g., "the", "a"), move it to the next segment:
                    adjusted_end -= 1
                    if i + 1 < len(boundaries):
                        next_start, next_end = boundaries[i + 1]
                        boundaries[i + 1] = (adjusted_end, next_end)  # Move next segment's start back
                    if verbose:
                        print(f"   Segment {i}: Moved determiner '{tokens[adjusted_end]}' to next segment")

            
            # Look at the START of current segment - is there a sentence ending we should move to previous?
            if i > 0:  # Not the first segment
                # Search backward for sentence ending
                for look_pos in range(start-1, start - max_look_distance - 1, -1):
                    if look_pos > 0 and tokens[look_pos] in endings:
                        # Found a sentence ending nearby - move it to previous segment
                        adjusted_start = look_pos + 1  # Start after the sentence ending
                        
                        # Update previous segment to include this sentence ending
                        if adjusted_boundaries:
                            prev_start, prev_end = adjusted_boundaries[-1]
                            adjusted_boundaries[-1] = (prev_start, adjusted_start)
                            
                            if verbose:
                                print(f"   Segment {i}: Moved start from {start} to {adjusted_start} (moved '{tokens[look_pos]}' to segment {i-1})")
                        break
            
            adjusted_boundaries.append((adjusted_start, adjusted_end))
        
        return adjusted_boundaries
    
    def align_segments_proportional(
        self, 
        source_segments: List[str], 
        translated_text: str,
        realign_on_sentences: bool = True,
        max_look_distance: int = 3,
        verbose: bool = False
    ) -> List[AlignedSegment]:
        """
        Proportionally allocate target tokens with smart punctuation handling.
        
        Steps:
        1. Tokenize and separate words from punctuation
        2. Allocate based on word count only
        3. Attach punctuation to previous word segments
        4. Optionally realign on sentence boundaries
        
        Args:
            source_segments: List of source text segments
            translated_text: Full translated text
            realign_on_sentences: If True, merge segments to end at sentence boundaries
            verbose: Print debug information
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 Proportional Alignment: {len(source_segments)} segments")
            print(f"{'='*70}")
        
        # Tokenize source segments and count words only
        source_word_counts = []
        
        for seg in source_segments:
            tokens = self.tokenize(seg)
            word_count = sum(1 for t in tokens if not self._is_punctuation(t))
            source_word_counts.append(word_count)
        
        total_source_words = sum(source_word_counts)
        
        # Tokenize target and separate words from punctuation
        target_tokens = self.tokenize(translated_text)
        target_word_indices, target_punct_indices = self._separate_words_and_punctuation(target_tokens)
        total_target_words = len(target_word_indices)
        target_lang = self._detect_lang(translated_text)
        
        if verbose:
            print(f"📝 Source: {total_source_words} words (excluding punctuation)")
            print(f"📝 Target: {total_target_words} words, {len(target_punct_indices)} punctuation (lang: {target_lang})")
            print(f"📊 Source word counts per segment: {source_word_counts}")
        
        # Calculate proportional allocation based on words only
        if total_source_words == 0:
            words_per_segment = [total_target_words // len(source_segments)] * len(source_segments)
            remainder = total_target_words % len(source_segments)
            for i in range(remainder):
                words_per_segment[i] += 1
        else:
            words_per_segment = []
            for word_count in source_word_counts:
                proportion = word_count / total_source_words
                allocated = max(1, int(round(proportion * total_target_words)))
                words_per_segment.append(allocated)
        
        # Adjust to match total target words exactly
        diff = total_target_words - sum(words_per_segment)
        if diff > 0:
            for i in range(diff):
                words_per_segment[i % len(words_per_segment)] += 1
        elif diff < 0:
            for i in range(abs(diff)):
                max_idx = max((idx for idx, val in enumerate(words_per_segment) if val > 1), 
                             key=lambda idx: words_per_segment[idx], 
                             default=0)
                words_per_segment[max_idx] -= 1
        
        if verbose:
            print(f"📊 Allocated words per segment: {words_per_segment} (sum={sum(words_per_segment)})")
        
        # Create boundaries based on word indices only
        word_boundaries = []
        word_pos = 0
        
        for word_count in words_per_segment:
            start_idx = target_word_indices[word_pos] if word_pos < len(target_word_indices) else len(target_tokens)
            end_pos = min(word_pos + word_count, len(target_word_indices))
            end_idx = target_word_indices[end_pos - 1] + 1 if end_pos > 0 else start_idx
            
            word_boundaries.append((start_idx, end_idx))
            word_pos = end_pos
        
        if verbose:
            print(f"📍 Word-based boundaries: {word_boundaries}")
        
        # Attach punctuation to previous word segments
        boundaries_with_punct = self._attach_punctuation_to_words(target_tokens, word_boundaries)
        
        if verbose:
            print(f"📍 After attaching punctuation: {boundaries_with_punct}")
        
        # Optional: Realign on sentence boundaries
        if realign_on_sentences:
            final_boundaries = self._realign_on_sentence_boundaries(
                target_tokens, boundaries_with_punct, target_lang, verbose, max_look_distance
            )
        else:
            final_boundaries = boundaries_with_punct
        
        if verbose:
            print(f"📍 Final boundaries: {final_boundaries}")
        
        # Create aligned segments
        aligned_segments = []
        
        for i, (start_pos, end_pos) in enumerate(final_boundaries):
            # Skip empty segments
            if end_pos - start_pos ==0:
                continue

            segment_tokens = target_tokens[start_pos:end_pos]
            translated_seg = self._reconstruct_text(segment_tokens)

            source_indices = [min(i, len(source_segments) - 1)]
            if i + 1 < len(final_boundaries) and final_boundaries[i+1][1] - final_boundaries[i+1][0] == 0:
                source_indices.append(min(i + 1, len(source_segments) - 1))

            # this will only be triggered in the pass of second target segment if the first target segment was empty
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
            
            if verbose:
                token_count = end_pos - start_pos
                first_token = segment_tokens[0] if segment_tokens else ""
                last_token = segment_tokens[-1] if segment_tokens else ""
                print(f"[{i}] {original_text[:40]:40s} → {translated_seg[:50]:50s}")
                print(f"     ({token_count} tokens: [{start_pos}:{end_pos}], starts:'{first_token}', ends:'{last_token}')")
        
        if verbose:
            print(f"✅ Done: {len(aligned_segments)} segments")
            print(f"{'='*70}\n")
        
        return aligned_segments