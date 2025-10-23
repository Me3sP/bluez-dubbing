import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple
from pathlib import Path
import shlex
from common_schemas.models import Word, SubtitleSegment
import subprocess


class SegmentCopySubtitleBuilder:
    """
    Simple and robust subtitle builder:
    - Use data['segments'] as ground truth (start, end, text).
    - Copy text as-is into cues.
    - If text is too long, wrap into 1-2 lines per cue up to max_chars_per_line.
    - If still too long for 2 lines, split into multiple cues with proportional timing.
    - Never drop words or punctuation.
    """
    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        min_duration: float = 0.7,
        max_duration: float = 7.0,
        mobile_mode: bool = True,
    ):
        if mobile_mode:
            max_chars_per_line = 30
            max_lines = 2  # keep 2 lines for readability on mobile
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.max_chars = max_chars_per_line * max_lines
        self.min_duration = min_duration
        self.max_duration = max_duration

    def build_from_segments(self, segments_json: List[dict]) -> List[SubtitleSegment]:
        out: List[SubtitleSegment] = []
        for seg in segments_json:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start = float(seg["start"])
            end = float(seg["end"])
            duration = max(0.01, end - start)

            chunks = self._chunk_text(text)

            # Fast path: short text fits in a single cue
            if len(chunks) == 1 and sum(len(line) for line in chunks[0]) <= self.max_chars:
                out.append(SubtitleSegment(
                    start=start, end=end, text="\n".join(chunks[0]), lines=chunks[0]
                ))
                continue

            # Allocate time proportionally across chunks by character count
            char_counts = [sum(len(line) for line in c) for c in chunks]
            total_chars = max(1, sum(char_counts))

            durations = [max(self.min_duration, duration * (cc / total_chars)) for cc in char_counts]
            sum_durs = sum(durations)
            if sum_durs > duration:
                scale = duration / sum_durs
                durations = [max(0.01, d * scale) for d in durations]

            t = start
            for lines, d in zip(chunks, durations):
                t_end = min(end, t + max(0.01, d))
                out.append(SubtitleSegment(
                    start=t, end=t_end, text="\n".join(lines), lines=lines
                ))
                t = t_end

            # Ensure last chunk ends exactly at the original end
            if out and out[-1].end < end - 1e-3:
                out[-1].end = end

        return out

    def _chunk_text(self, text: str) -> List[List[str]]:
        """
        Break text into chunks; each chunk has 1-2 lines within max_chars_per_line.
        """
        words = text.split()
        chunks: List[List[str]] = []
        i = 0

        while i < len(words):
            line1: List[str] = []
            line2: List[str] = []

            # Fill line 1
            while i < len(words):
                candidate = (" ".join(line1 + [words[i]])).strip()
                if len(candidate) <= self.max_chars_per_line or not line1:
                    if len(candidate) > self.max_chars_per_line and line1:
                        break
                    line1.append(words[i])
                    i += 1
                else:
                    break

            # Fill line 2 (if allowed)
            if self.max_lines > 1 and i < len(words):
                while i < len(words):
                    candidate = (" ".join(line2 + [words[i]])).strip()
                    if len(candidate) <= self.max_chars_per_line or not line2:
                        if len(candidate) > self.max_chars_per_line and line2:
                            break
                        line2.append(words[i])
                        i += 1
                    else:
                        break
                # Balance lines for nicer layout
                line1, line2 = self._balance_lines(line1, line2)

            lines = [" ".join(line1)]
            if line2:
                lines.append(" ".join(line2))
            chunks.append(lines)

        return chunks

    def _balance_lines(self, l1: List[str], l2: List[str]) -> Tuple[List[str], List[str]]:
        """
        Move last word(s) from line 1 to line 2 to balance lengths.
        """
        def L(s: List[str]) -> int:
            return len(" ".join(s)) if s else 0

        s1, s2 = L(l1), L(l2)
        while l2 is not None and (s1 - s2) > 6 and len(l1) > 1:
            w = l1.pop()
            l2.insert(0, w)
            s1, s2 = L(l1), L(l2)
        return l1, l2


def build_subtitles_from_asr_result(
    data: Path | str | dict | List[dict],
    output_dir: Path | str,
    custom_name: Optional[str] = None,
    formats: List[str] = ["srt", "vtt"],
    mobile_mode: bool = True,
) -> List[str]:
    # Load JSON if a path is provided
    if isinstance(data, (str, Path)):
        with open(data, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # If a full ASR dict with 'segments' is provided, use the simple builder
    segments: List[SubtitleSegment]
    if isinstance(data, dict) and "segments" in data:
        seg_json = [s for s in data["segments"] if s and s.get("text")]
        builder = SegmentCopySubtitleBuilder(mobile_mode=mobile_mode)
        segments = builder.build_from_segments(seg_json)
    elif isinstance(data, list):
        # Assume list of segments in dict form
        seg_json = [s for s in data if s and s.get("text")]
        builder = SegmentCopySubtitleBuilder(mobile_mode=mobile_mode)
        segments = builder.build_from_segments(seg_json)
    else:
        raise ValueError("Invalid data format for subtitles generation.")

    # Write outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = custom_name if custom_name else "result"
    suffix = "_mobile" if mobile_mode else ""

    out_paths = []
    for fmt in formats:
        out_paths.append(write_subtitles(
            segments,
            output_dir / f"{name}{suffix}_subtitles",
            format=fmt
        ))

    # Stats
    print(f"\n📊 Subtitle Statistics ({'Mobile' if mobile_mode else 'Desktop'} Mode):")
    print(f"   Total segments: {len(segments)}")
    if segments:
        print(f"   Avg duration: {sum(s.duration for s in segments) / len(segments):.2f}s")
        print(f"   Avg CPS: {sum(s.cps for s in segments) / len(segments):.2f}")
        print(f"   Avg chars/subtitle: {sum(s.char_count for s in segments) / len(segments):.0f}")

    return out_paths


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