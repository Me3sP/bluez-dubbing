import subprocess
from pathlib import Path
from .subtitles_handling import burn_subtitles_to_video, SubtitleStyle
from .audio_processing import get_audio_duration
from typing import Optional


def apply_audio_to_video(
    video_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    dubbing_strategy: str = "default",
    orig_duck: float = 0.2,
) -> Path:
    """
    Replace or overlay audio on a video.
    - default: replace original audio with TTS (pad and trim to video duration).
    - translation_over: overlay TTS as voice-over and duck original audio when TTS speaks.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    vd = get_audio_duration(video_path)  # video duration in seconds (ms precision)

    if dubbing_strategy == "translation_over":  # translation_over
        # Inputs:
        #  - 0:v video
        #  - 0:a original audio
        #  - 1:a TTS (voice-over)
        #
        # Steps:
        #  - Ensure both audios are exactly video length.
        #  - Duck original with TTS as sidechain.
        #  - Mix ducked original + TTS.
        #
        # Notes:
        #  - sidechaincompress reduces original's gain only when TTS is present.
        #  - Adjust ratio/threshold/attack/release to taste.
        
        filter_complex = (
            f"[1:a]apad[padded];"
            f"[padded]atrim=0:{vd:.3f},asetpts=PTS-STARTPTS[voice];"
            f"[0:a]atrim=0:{vd:.3f},asetpts=PTS-STARTPTS,volume={orig_duck}[orig];"
            f"[orig][voice]amix=inputs=2:weights=1|1:normalize=0,aresample=async=1:first_pts=0[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]
        
    else : # default: full_replacement
        # full_replacement: replace original audio with TTS, pad and trim to exact video duration
        filter_complex = f"[1:a]apad,atrim=0:{vd:.3f},asetpts=PTS-STARTPTS[aout]"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    

    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    

def final(
    video_path: Path | str,
    audio_path: Path | str,
    dubbed_path: Path | str,
    output_path: Path | str,
    subtitle_path: Path | str | None = None,
    sub_style: Optional[SubtitleStyle] = None,
    mobile_optimized: bool = False,
    dubbing_strategy: str = "default",
    orig_duck: float = 0.2,
) -> None:
    """
    Replace video's audio stream with dubbed audio or add voice-over, then burn subtitles.
    """
    if dubbed_path != video_path:
        apply_audio_to_video(
            video_path=video_path,
            audio_path=audio_path,
            output_path=dubbed_path,
            dubbing_strategy=dubbing_strategy,
            orig_duck=orig_duck,
        )

    if sub_style is not None and subtitle_path is not None:
        burn_subtitles_to_video(
        video_path=dubbed_path,  # dubbed video
        subtitle_path=subtitle_path,  # SRT/ASS file
        output_path=output_path,  # Final output path
        style=sub_style,
        mobile=mobile_optimized
    )