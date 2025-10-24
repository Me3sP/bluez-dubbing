from audio_separator.separator import Separator
from pathlib import Path
import os
import subprocess
from typing import Dict, Any, List, Optional

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
# ===================================== audio separator utils =====================================

# Perform source separation and return the path to the instrumental audio file
def separation(input_file: str, output_dir: str, model_filename: str, output_format: str, custom_output_names: dict,  model_file_dir: str = "/tmp/audio-separator-models/"):
    separator = Separator(output_format=output_format, output_dir=output_dir, model_file_dir=model_file_dir)
    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(input_file, custom_output_names=custom_output_names)
    return output_files


def filter_supported_models_grouped(stems_count: int = 2, contains: str = "vocals") -> Dict:
    """
    Keep models grouped by architecture and return a list of {'filename', 'stems'} per arch.
    Filters by exact number of stems and presence of a keyword in stems.
    """
    separator = Separator()
    models_by_type = separator.list_supported_model_files()

    kw = (contains or "").lower()

    def to_stem_list(stems) -> List[str]:
        if stems is None:
            return []
        if isinstance(stems, dict):
            return list(stems.keys())
        if isinstance(stems, (list, tuple)):
            return [str(s) for s in stems]
        return [str(stems)]

    out: Dict[str, List[Dict[str, List[str]]]] = {}

    for arch, models in (models_by_type or {}).items():
        bucket: List[Dict[str, List[str]]] = []

        # Normalize iteration over models: dict(name -> meta) or list of meta
        if isinstance(models, dict):
            iterable = models.values()
        elif isinstance(models, list):
            iterable = models
        else:
            iterable = []

        for meta in iterable:
            if not isinstance(meta, dict):
                continue
            stems_list = to_stem_list(meta.get("stems"))
            if stems_count is not None and len(stems_list) != stems_count:
                continue
            if kw and not any(kw in str(s).lower() for s in stems_list):
                continue
            filename = meta.get("filename")
            if filename:
                bucket.append({"filename": filename, "stems": stems_list})

        if bucket:
            out[arch] = bucket

    return out


def _to_stem_list(stems) -> List[str]:
    if stems is None:
        return []
    if isinstance(stems, dict):
        return list(stems.keys())
    if isinstance(stems, (list, tuple)):
        return [str(s) for s in stems]
    return [str(stems)]

def get_non_vocals_stem(model_filename: str, vocals_label: str = "vocals") -> Optional[str]:
    """
    Given the grouped output of separator.list_supported_model_files() (or the filtered result),
    return the stem name that is not 'vocals' for the specified model filename.
    """
    separator = Separator()
    models_by_type = separator.list_supported_model_files()

    target = str(model_filename).strip()
    v_kw = vocals_label.lower().strip()

    # Iterate across architectures
    for _, models in (models_by_type or {}).items():
        # Normalize iteration over models collections
        if isinstance(models, dict):
            iterable = models.values()
        elif isinstance(models, list):
            iterable = models
        else:
            iterable = []

        for meta in iterable:
            if not isinstance(meta, dict):
                continue
            fname = str(meta.get("filename", "")).strip()
            if fname != target:
                continue
            stems_list = _to_stem_list(meta.get("stems"))
            for s in stems_list:
                if s is None:
                    continue
                if str(s).lower().strip() != v_kw:
                    return str(s)
            return None  # Found model but no non-vocals stem

    return None  # Model filename not found

# Example:
# grouped_all = separator.list_supported_model_files()
# get_non_vocals_stem(grouped_all, "1_HP-UVR.pth")
# get_non_vocals_stem(two_stem_vocals_by_arch, "1_HP-UVR.pth")
# ...existing code...