from audio_separator.separator import Separator
from pathlib import Path
import os
import subprocess


# output_base = Path(__file__).resolve().parents[4]
# OUT = output_base / "outs" / "preprocessing_outputs"



def convert_video_to_audio(video_file: str, audio_dir: str):
    os.makedirs(audio_dir, exist_ok=True)
    video_base = os.path.basename(video_file).rsplit('.', 1)[0]
    _RAW_AUDIO_FILE = os.path.join(audio_dir, f"{video_base}_raw_audio.mp3")
    if not os.path.exists(_RAW_AUDIO_FILE):
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn',
            '-c:a', 'libmp3lame', '-b:a', '32k',
            '-ar', '16000',
            '-ac', '1', 
            '-metadata', 'encoding=UTF-8', _RAW_AUDIO_FILE
        ], check=True, stderr=subprocess.PIPE)

    return _RAW_AUDIO_FILE

# Perform source separation and return the path to the instrumental audio file
def separation(input_file: str, output_dir: str, model_filename: str, output_format: str, custom_output_names: dict,  model_file_dir: str = "/tmp/audio-separator-models/") -> str:
    separator = Separator(output_format=output_format, output_dir=output_dir, model_file_dir=model_file_dir)
    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(input_file, custom_output_names=custom_output_names)
    return output_files[0] if output_files else ""