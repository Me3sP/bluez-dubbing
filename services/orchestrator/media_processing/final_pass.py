import subprocess
from pathlib import Path


def apply_audio_to_video(video_path: str, audio_path: Path, subtitle_path: Path, output_path: Path):
    """
    Replace video's audio stream with dubbed audio and burn subtitles.
    """
    # FFmpeg command to replace audio and burn subtitles
    subprocess.run([
        'ffmpeg', '-y',
        '-i', video_path,           # Input video
        '-i', str(audio_path),       # New audio
        '-vf', f"subtitles={subtitle_path}",  # Burn subtitles
        '-map', '0:v:0',             # Video from first input
        '-map', '1:a:0',             # Audio from second input
        '-c:v', 'libx264',           # Video codec
        '-c:a', 'aac',               # Audio codec
        '-b:a', '192k',
        '-shortest',                 # Match shortest stream
        str(output_path)
    ], check=True, stderr=subprocess.PIPE)