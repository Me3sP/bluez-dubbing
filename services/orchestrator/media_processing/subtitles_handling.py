import json
from pathlib import Path

# ---------- Subtitle writers ----------
def srt_timestamp(t):
    h = int(t // 3600); t -= 3600*h
    m = int(t // 60);   t -= 60*m
    s = int(t);         ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def ts(t):
    h = int(t // 3600); t -= 3600*h
    m = int(t // 60);   t -= 60*m
    s = int(t);         ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def write_srt(chunks, path):
    # Handle if chunks is a path to JSON file
    if isinstance(chunks, (str, Path)):
        with open(chunks, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, 1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(c['start'])} --> {srt_timestamp(c['end'])}\n")
            f.write(c["text"] + "\n\n")

def write_vtt(chunks, path):
    # Handle if chunks is a path to JSON file
    if isinstance(chunks, (str, Path)):
        with open(chunks, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for c in chunks:
            f.write(f"{ts(c['start'])} --> {ts(c['end'])}\n")
            f.write(c["text"] + "\n\n")
