import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Literal
from pathlib import Path
import shlex
from common_schemas.models import Word, SubtitleSegment
import subprocess

class ProfessionalSubtitleBuilder:
    """
    Builds professional-grade subtitles from word segments.
    
    Standards:
    - Max 2 lines per subtitle
    - Max 42 characters per line
    - Max reading speed: 17-20 CPS (characters per second)
    - Min duration: 0.7 seconds
    - Max duration: 7 seconds
    - Min gap between subtitles: 0.1 seconds
    - Respect sentence boundaries
    - Balance line lengths
    """
    
    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        max_cps: float = 20.0,
        min_cps: float = 5.0,
        min_duration: float = 0.7,
        max_duration: float = 7.0,
        min_gap: float = 0.1,
        hard_break_chars: str = ".!?",
        soft_break_chars: str = ",;:",
        mobile_mode: bool = False,  # NEW: Mobile optimization
    ):
        # Apply mobile-specific constraints
        if mobile_mode:
            self.max_chars_per_line = 30  # Shorter lines for mobile
            self.max_lines = 1  # Prefer single line
            self.max_chars = 30
            self.max_cps = 18.0  # Slightly slower for readability
        else:
            self.max_chars_per_line = max_chars_per_line
            self.max_lines = max_lines
            self.max_chars = max_chars_per_line * max_lines
            self.max_cps = max_cps
        
        self.min_cps = min_cps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_gap = min_gap
        self.hard_break_chars = hard_break_chars
        self.soft_break_chars = soft_break_chars
        self.mobile_mode = mobile_mode

    
    def build_subtitles(self, words: List[Word]) -> List[SubtitleSegment]:
        """
        Build subtitle segments from word list.
        
        Args:
            words: List of Word objects with timing information
            
        Returns:
            List of SubtitleSegment objects
        """
        if not words:
            return []
        
        segments = []
        current_words = []
        
        for i, word in enumerate(words):
            current_words.append(word)
            
            # Check if we should break here
            should_break = self._should_break(
                current_words, 
                word, 
                i < len(words) - 1 and words[i + 1] or None
            )
            
            if should_break or i == len(words) - 1:
                segment = self._create_segment(current_words)
                if segment:
                    segments.append(segment)
                current_words = []
        
        # Post-process: merge very short segments and split very long ones
        segments = self._post_process_segments(segments)
        
        return segments
    
    def _should_break(
        self, 
        current_words: List[Word], 
        current_word: Word,
        next_word: Optional[Word]
    ) -> bool:
        """Determine if we should break the subtitle here."""
        if not current_words:
            return False
        
        # Calculate current segment stats
        text = " ".join(w.text for w in current_words)
        start = current_words[0].start
        end = current_words[-1].end
        duration = end - start
        char_count = len(text)
        cps = char_count / duration if duration > 0 else 0
        
        # 1. Hard break: sentence ending punctuation
        if current_word.text.rstrip()[-1:] in self.hard_break_chars:
            # Check if there's a significant pause after
            if next_word and (next_word.start - end) > 0.3:
                return True
        
        # 2. Character limit exceeded
        if char_count >= self.max_chars:
            return True
        
        # 3. Reading speed too fast
        if cps > self.max_cps and len(current_words) > 3:
            return True
        
        # 4. Duration limits
        if duration >= self.max_duration:
            return True
        
        # 5. Soft break: comma or semicolon with pause
        if current_word.text.rstrip()[-1:] in self.soft_break_chars:
            if next_word and (next_word.start - end) > 0.2:
                if char_count >= self.max_chars * 0.6:  # At least 60% full
                    return True
        
        # 6. Natural pause between words
        if next_word:
            gap = next_word.start - end
            if gap > 0.5 and char_count >= self.max_chars * 0.5:
                return True
        
        return False
    
    def _create_segment(self, words: List[Word]) -> Optional[SubtitleSegment]:
        """Create a subtitle segment from words."""
        if not words:
            return None
        
        text = " ".join(w.text for w in words)
        start = words[0].start
        end = words[-1].end
        duration = end - start
        
        # Skip if too short
        if duration < self.min_duration and len(words) < 2:
            return None
        
        # Split into lines
        lines = self._split_into_lines(text)
        
        return SubtitleSegment(
            start=start,
            end=end,
            text=text,
            lines=lines
        )
    
    def _split_into_lines(self, text: str) -> List[str]:
        """Split text into balanced lines respecting word boundaries."""
        if len(text) <= self.max_chars_per_line:
            return [text]
        
        # Mobile mode: prefer single line with abbreviations
        if self.mobile_mode:
            return self._mobile_optimize_text(text)
        
        # Try to split at natural break points
        words = text.split()
        
        if len(words) == 1:
            # Single long word - keep as is
            return [text]
        
        # Find best split point
        mid_point = len(text) // 2
        best_split = 0
        min_diff = float('inf')
        
        current_len = 0
        for i, word in enumerate(words[:-1]):
            current_len += len(word) + 1  # +1 for space
            diff = abs(current_len - mid_point)
            
            # Check if both lines would be within limit
            line1 = " ".join(words[:i+1])
            line2 = " ".join(words[i+1:])
            
            if (len(line1) <= self.max_chars_per_line and 
                len(line2) <= self.max_chars_per_line and 
                diff < min_diff):
                min_diff = diff
                best_split = i + 1
        
        if best_split > 0:
            line1 = " ".join(words[:best_split])
            line2 = " ".join(words[best_split:])
            return [line1, line2]
        
        # Fallback: force split at character limit
        return [text[:self.max_chars_per_line], text[self.max_chars_per_line:]]
    
    def _post_process_segments(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Post-process segments to fix issues."""
        if not segments:
            return []
        
        processed = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # Check if segment is too short
            if current.duration < self.min_duration and i < len(segments) - 1:
                next_seg = segments[i + 1]
                
                # Try to merge with next if gap is small
                gap = next_seg.start - current.end
                combined_text = f"{current.text} {next_seg.text}"
                
                if (gap < self.min_gap and 
                    len(combined_text) <= self.max_chars):
                    # Merge segments
                    merged = SubtitleSegment(
                        start=current.start,
                        end=next_seg.end,
                        text=combined_text,
                        lines=self._split_into_lines(combined_text)
                    )
                    processed.append(merged)
                    i += 2
                    continue
            
            # Check if reading speed is too slow
            if current.cps < self.min_cps and current.duration > 2.0:
                # Shorten duration
                target_duration = current.char_count / self.min_cps
                current.end = current.start + target_duration
            
            processed.append(current)
            i += 1
        
        return processed
    
    def _mobile_optimize_text(self, text: str) -> List[str]:
        """Optimize text for mobile by abbreviating if needed."""
        if len(text) <= self.max_chars_per_line:
            return [text]
        
        # Try common abbreviations
        abbreviations = {
            "and": "&",
            "you": "u",
            "are": "r",
            "with": "w/",
            "without": "w/o",
            "because": "bc",
            "before": "b4",
        }
        
        words = text.split()
        abbreviated = []
        
        for word in words:
            lower_word = word.lower().strip('.,!?;:')
            if lower_word in abbreviations:
                # Preserve punctuation
                punct = ''.join(c for c in word if c in '.,!?;:')
                abbreviated.append(abbreviations[lower_word] + punct)
            else:
                abbreviated.append(word)
        
        abbreviated_text = " ".join(abbreviated)
        
        # # If still too long, truncate with ellipsis
        # if len(abbreviated_text) > self.max_chars_per_line:
        #     return [abbreviated_text[:self.max_chars_per_line - 3] + "..."]
        
        return [abbreviated_text]


def format_timestamp(seconds: float, format: str = "srt") -> str:
    """Format timestamp for subtitle files."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if format == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    elif format == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        raise ValueError(f"Unsupported format: {format}")


def segments_to_srt(segments: List[SubtitleSegment]) -> str:
    """Convert segments to SRT format."""
    lines = []
    
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg.start, 'srt')} --> {format_timestamp(seg.end, 'srt')}")
        lines.extend(seg.lines)
        lines.append("")  # Empty line between segments
    
    return "\n".join(lines)


def segments_to_vtt(segments: List[SubtitleSegment]) -> str:
    """Convert segments to WebVTT format."""
    lines = ["WEBVTT", ""]
    
    for seg in segments:
        lines.append(f"{format_timestamp(seg.start, 'vtt')} --> {format_timestamp(seg.end, 'vtt')}")
        lines.extend(seg.lines)
        lines.append("")
    
    return "\n".join(lines)


def write_subtitles(
    segments: List[SubtitleSegment],
    output_path: Path,
    format: str = "srt"
) -> str:
    """Write subtitles to file."""
    output_path = Path(output_path)
    
    if format == "srt":
        content = segments_to_srt(segments)
        output_path = output_path.with_suffix(".srt")
    elif format == "vtt":
        content = segments_to_vtt(segments)
        output_path = output_path.with_suffix(".vtt")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    output_path.write_text(content, encoding="utf-8")
    print(f"✅ Wrote {len(segments)} subtitle segments to {output_path}")

    return str(output_path)


# Example usage function
def build_subtitles_from_asr_result(
    data: Path | str | dict | List[dict],
    output_dir: Path | str,
    custom_name: Optional[str] = None,
    formats: List[str] = ["srt", "vtt"],
    mobile_mode: bool = True,  # NEW
) -> List[str]:
    """
    Build professional subtitles from ASR result JSON.
    
    Args:
        data: ASR result (path, dict, or list of words)
        output_dir: Directory to save subtitle files
        custom_name: Custom base name for output files
        formats: List of formats to generate ("srt", "vtt", "ass")
        mobile_mode: Optimize for mobile screens
    """
    
    if isinstance(data, (str, Path)):
        with open(data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Convert to Word objects
    if isinstance(data, list):
        words = data
    else:
        words = [
            Word(
                text=w['text'],
                start=w['start'],
                end=w['end'],
                score=w.get('score'),
                speaker_id=w.get('speaker_id')
            )
            for w in data.get('WordSegments', [])
        ]
    
    # Build subtitle segments with mobile optimization
    builder = ProfessionalSubtitleBuilder(mobile_mode=mobile_mode)
    segments = builder.build_subtitles(words)
    
    # Write to files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = custom_name if custom_name else "result"
    suffix = "_mobile" if mobile_mode else ""

    out = []
    for fmt in formats:
        out.append(write_subtitles(
            segments, 
            output_dir / f"{name}{suffix}_subtitles", 
            format=fmt
        ))
    
    # Print statistics
    print(f"\n📊 Subtitle Statistics ({('Mobile' if mobile_mode else 'Desktop')} Mode):")
    print(f"   Total segments: {len(segments)}")
    if segments:
        print(f"   Avg duration: {sum(s.duration for s in segments) / len(segments):.2f}s")
        print(f"   Avg CPS: {sum(s.cps for s in segments) / len(segments):.2f}")
        print(f"   Avg chars/subtitle: {sum(s.char_count for s in segments) / len(segments):.0f}")

    return out


@dataclass
class SubtitleStyle:
    """Subtitle appearance configuration."""
    # Font settings
    font_name: str = "Arial"
    font_size: int = 24
    font_color: str = "white"  # HTML color or &HBBGGRR
    bold: bool = True
    italic: bool = False
    
    # Outline and shadow
    outline_color: str = "black"
    outline_width: int = 2
    shadow_offset: int = 2
    
    # Background
    background_color: Optional[str] = None  # None = transparent, or "black@0.5" for semi-transparent
    
    # Position
    alignment: Literal["bottom", "top", "center"] = "bottom"
    margin_v: int = 20  # Vertical margin from edge (pixels)
    margin_h: int = 20  # Horizontal margin from edge
    
    # Mobile-specific overrides
    mobile_font_size: int = 18
    mobile_margin_v: int = 40  # More space on mobile
    
    def to_ass_style(self, mobile: bool = False) -> str:
        """Convert to ASS subtitle format style string."""
        font_size = self.mobile_font_size if mobile else self.font_size
        margin_v = self.mobile_margin_v if mobile else self.margin_v
        
        # ASS alignment codes: 1=left-bottom, 2=center-bottom, 3=right-bottom
        # 4=left-middle, 5=center-middle, 6=right-middle
        # 7=left-top, 8=center-top, 9=right-top
        alignment_map = {
            "bottom": 2,  # Center bottom
            "top": 8,     # Center top
            "center": 5,  # Center middle
        }
        alignment_code = alignment_map[self.alignment]
        
        # Convert colors to ASS format (&HAABBGGRR)
        font_color_ass = self._html_to_ass_color(self.font_color)
        outline_color_ass = self._html_to_ass_color(self.outline_color)
        
        # Background color
        if self.background_color:
            bg_color_ass = self._html_to_ass_color(self.background_color)
        else:
            bg_color_ass = "&H00000000"  # Transparent
        
        return (
            f"FontName={self.font_name},"
            f"FontSize={font_size},"
            f"PrimaryColour={font_color_ass},"
            f"OutlineColour={outline_color_ass},"
            f"BackColour={bg_color_ass},"
            f"Bold={'1' if self.bold else '0'},"
            f"Italic={'1' if self.italic else '0'},"
            f"BorderStyle=1,"
            f"Outline={self.outline_width},"
            f"Shadow={self.shadow_offset},"
            f"Alignment={alignment_code},"
            f"MarginL={self.margin_h},"
            f"MarginR={self.margin_h},"
            f"MarginV={margin_v}"
        )
    
    def _html_to_ass_color(self, color: str) -> str:
        """Convert HTML color to ASS format."""
        if color.startswith("#"):
            # #RRGGBB -> &H00BBGGRR
            r, g, b = color[1:3], color[3:5], color[5:7]
            return f"&H00{b}{g}{r}".upper()
        elif "@" in color:
            # "black@0.5" -> color with alpha
            color_part, alpha = color.split("@")
            alpha_hex = format(int((1 - float(alpha)) * 255), '02X')
            if color_part == "black":
                return f"&H{alpha_hex}000000"
            elif color_part == "white":
                return f"&H{alpha_hex}FFFFFF"
        
        # Named colors
        color_map = {
            "white": "&H00FFFFFF",
            "black": "&H00000000",
            "yellow": "&H0000FFFF",
            "red": "&H000000FF",
            "blue": "&H00FF0000",
            "green": "&H0000FF00",
        }
        return color_map.get(color.lower(), "&H00FFFFFF")


def convert_srt_to_ass(
    srt_path: Path | str,
    output_path: Path | str,
    style: SubtitleStyle,
    video_width: int = 1920,
    video_height: int = 1080,
    mobile: bool = False
) -> Path:
    """
    Convert SRT to ASS format with custom styling.
    
    Args:
        srt_path: Input SRT file
        output_path: Output ASS file
        style: SubtitleStyle configuration
        video_width: Video width (for positioning)
        video_height: Video height (for positioning)
        mobile: Use mobile-optimized settings
    
    Returns:
        Path to generated ASS file
    """
    output_path = Path(output_path).with_suffix(".ass")
    
    # Read SRT content
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    # Create ASS header
    ass_header = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, {style.to_ass_style(mobile).replace(',', ', ')}
Style: Default,{style.to_ass_style(mobile)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Parse SRT and convert to ASS events
    events = []
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # Parse timing
        timing_line = lines[1]
        start, end = timing_line.split(' --> ')
        start_ass = _srt_time_to_ass(start)
        end_ass = _srt_time_to_ass(end)
        
        # Parse text
        text = '\n'.join(lines[2:])
        text_ass = text.replace('\n', '\\N')  # ASS line break
        
        events.append(
            f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text_ass}"
        )
    
    # Write ASS file
    ass_content = ass_header + '\n'.join(events)
    output_path.write_text(ass_content, encoding='utf-8')
    
    print(f"✅ Converted SRT to ASS: {output_path}")
    return output_path


def _srt_time_to_ass(srt_time: str) -> str:
    """Convert SRT timestamp to ASS format."""
    # SRT: 00:00:01,500 -> ASS: 0:00:01.50
    time_part, ms = srt_time.strip().split(',')
    h, m, s = time_part.split(':')
    return f"{int(h)}:{m}:{s}.{ms[:2]}"

def _hex_to_ass_colour(rgb_hex: str) -> str:
    # Input: "#RRGGBB" => Output: &HAABBGGRR (AA=00 opaque)
    h = rgb_hex.lstrip("#")
    if len(h) != 6:
        return "&H00FFFFFF"  # white fallback
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"

def _style_to_force_style(style: "SubtitleStyle", mobile: bool = False) -> str:
    parts = []
    if getattr(style, "font_name", None):
        parts.append(f"Fontname={style.font_name}")
    if getattr(style, "font_size", None):
        sz = style.mobile_font_size if mobile and getattr(style, "mobile_font_size", None) else style.font_size
        parts.append(f"Fontsize={int(sz)}")
    if getattr(style, "font_color", None) and style.font_color.startswith("#"):
        parts.append(f"PrimaryColour={_hex_to_ass_colour(style.font_color)}")
    if getattr(style, "outline_color", None) and style.outline_color.startswith("#"):
        parts.append(f"OutlineColour={_hex_to_ass_colour(style.outline_color)}")
    if getattr(style, "outline_width", None) is not None:
        parts.append(f"Outline={int(style.outline_width)}")
        # A small shadow helps readability when Outline>0
        if getattr(style, "shadow_offset", None) is not None:
            parts.append(f"Shadow={int(style.shadow_offset)}")
    if getattr(style, "bold", None) is not None:
        parts.append(f"Bold={1 if style.bold else 0}")
    if getattr(style, "italic", None) is not None:
        parts.append(f"Italic={1 if style.italic else 0}")
    mv = style.mobile_margin_v if mobile and getattr(style, "mobile_margin_v", None) else getattr(style, "margin_v", None)
    if mv is not None:
        parts.append(f"MarginV={int(mv)}")
    # Alignment: ASS codes (2=bottom-center, 8=top-center, 5=middle-center)
    align = getattr(style, "alignment", "bottom")
    if align == "top":
        parts.append("Alignment=8")
    elif align == "middle":
        parts.append("Alignment=5")
    else:
        parts.append("Alignment=2")
    return ",".join(parts)

def _ensure_ass(sub_path: Path) -> Path:
    """If not .ass, convert to .ass via ffmpeg and return new path."""
    if sub_path.suffix.lower() in [".ass", ".ssa"]:
        return sub_path
    out_path = sub_path.with_suffix(".ass")
    # Convert with ffmpeg demux/remux; ffmpeg will map cues appropriately
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(sub_path), str(out_path)],
        check=True, text=True, capture_output=True
    )
    return out_path


def burn_subtitles_to_video(
    video_path: Path | str,
    subtitle_path: Path | str,
    output_path: Path | str,
    style: SubtitleStyle | None = None,
    mobile: bool = False,
    fonts_dir: str | None = None,
) -> Path:
    """
    Burn subtitles into video using libass.
    - Converts SRT/VTT to ASS for styling.
    - Applies force_style derived from SubtitleStyle.
    """
    video = Path(video_path)
    subs = Path(subtitle_path)
    out = Path(output_path)
    ass_path = _ensure_ass(subs)

    force_style = _style_to_force_style(style, mobile) if style else ""
    # Build subtitles filter args
    # subtitles=path[:force_style=...][:fontsdir=...]
    sub_filter = f"subtitles={shlex.quote(str(ass_path))}"
    if force_style:
        sub_filter += f":force_style='{force_style}'"
    if fonts_dir:
        sub_filter += f":fontsdir={shlex.quote(fonts_dir)}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", sub_filter,
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out),
    ]
    print("🔥 Burning subtitles to video...")
    # Optional: print cmd for debugging
    # print("FFmpeg:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to burn subtitles:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    return out

# Apply custom styling
STYLE_PRESETS = {
    "default": SubtitleStyle(),
    "minimal": SubtitleStyle(
        font_size=20,
        mobile_font_size=16,
        outline_width=1,
        margin_v=10,
        mobile_margin_v=30
    ),
    "bold": SubtitleStyle(
        font_size=28,
        mobile_font_size=20,
        bold=True,
        outline_width=3,
        font_color="yellow"
    ),
    "netflix": SubtitleStyle(
        font_name="Netflix Sans",
        font_size=24,
        mobile_font_size=18,
        font_color="white",
        background_color="black@0.7",  # Semi-transparent black background
        outline_width=0,
        margin_v=40,
        mobile_margin_v=60
    ),
    "fantasy": SubtitleStyle(
        font_name="Uncial Antiqua",   # Use an installed fantasy-style font
        font_size=28,
        mobile_font_size=22,
        font_color="#FFD700",         # Gold
        outline_color="#3B0066",      # Deep purple outline
        outline_width=3,
        shadow_offset=4,
        background_color=None,        # No background box
        bold=False,
        italic=True,
        margin_v=50,
        mobile_margin_v=70,
        alignment="bottom",
    ),
}