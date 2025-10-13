import subprocess
from pathlib import Path
from .subtitles_handling import burn_subtitles_to_video, SubtitleStyle
from typing import Optional


def apply_audio_to_video(
    video_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str
) -> Path:
    """Replace video audio track with new audio."""
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    
    # Verify inputs exist
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    
    print(f"🎬 Replacing audio in video...")
    print(f"   Video: {video_path}")
    print(f"   Audio: {audio_path}")
    print(f"   Output: {output_path}")
    
    # Get video duration
    probe_video_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    video_duration = float(subprocess.check_output(probe_video_cmd, text=True).strip())
    
    # Get audio duration
    probe_audio_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]
    audio_duration = float(subprocess.check_output(probe_audio_cmd, text=True).strip())
    
    print(f"   Video duration: {video_duration:.2f}s")
    print(f"   Audio duration: {audio_duration:.2f}s")
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', str(video_path),
        '-i', str(audio_path),
        '-map', '0:v:0',  # Map video from first input
        '-map', '1:a:0',  # Map audio from second input
        '-c:v', 'copy',   # Copy video codec (CHANGED: don't re-encode)
        '-c:a', 'aac',    # Encode audio as AAC
        '-b:a', '192k',   # Audio bitrate
        '-shortest',      # End when shortest stream ends (CHANGED from -longest)
        str(output_path)
    ]
    
    # Run with output capture for better error messages
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Video with new audio: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")

def final(video_path: Path | str, audio_path: Path | str, dubbed_path: Path | str, subtitle_path: Path | str, output_path: Path | str, style: Optional[SubtitleStyle] = None, mobile_optimized: bool = False):
    """
    Replace video's audio stream with dubbed audio and burn subtitles.
    """

    apply_audio_to_video(
        video_path=video_path,
        audio_path=audio_path,
        output_path=dubbed_path
    )

    # # FFmpeg command to replace audio and burn subtitles
    # subprocess.run([
    #     'ffmpeg', '-y',
    #     '-i', video_path,           # Input video
    #     '-i', str(audio_path),       # New audio
    #     '-map', '0:v:0',             # Video from first input
    #     '-map', '1:a:0',             # Audio from second input
    #     '-c:v', 'libx264',           # Video codec
    #     '-c:a', 'aac',               # Audio codec
    #     '-b:a', '192k',
    #     '-longest',                  # Match longest stream
    #     str(dubbed_path)
    # ], check=True, stderr=subprocess.PIPE)

    return burn_subtitles_to_video(
        video_path=dubbed_path,  # Your dubbed video
        subtitle_path=subtitle_path,  # SRT file
        output_path=output_path,  # Final output path
        style=style,
        mobile=mobile_optimized
    )
