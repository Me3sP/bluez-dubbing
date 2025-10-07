import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import librosa
import soundfile as sf
import numpy as np
import subprocess
import json
from scipy import signal
import pyrubberband as prb
import os

def rubberband_to_duration(in_wav, target_ms, out_wav):
    """
    Adjust audio duration using Rubberband with high-quality settings.
    
    Args:
        in_wav: Input audio file path
        target_ms: Target duration in milliseconds
        out_wav: Output audio file path
    """
    # Read audio (always_2d=False for proper shape handling)


    # Support both string and Path for in_wav
    in_wav_path = str(in_wav) if isinstance(in_wav, Path) else in_wav
    y, sr = sf.read(in_wav_path, always_2d=False)
    
    # Calculate target samples and stretch rate
    target_samples = int(round(target_ms * sr / 1000))
    
    # Get current length (handle both mono and stereo)
    current_samples = y.shape[0] if y.ndim == 1 else y.shape[1]
    rate = current_samples / target_samples
    
    print(f"🎵 Rubberband time stretch:")
    print(f"   Current: {current_samples/sr:.3f}s ({current_samples} samples)")
    print(f"   Target: {target_ms/1000:.3f}s ({target_samples} samples)")
    print(f"   Rate: {rate:.3f}x")
    
    # Prepare audio for pyrubberband (expects channels-first for stereo)
    if y.ndim == 1:
        # Mono audio
        y_input = y
    else:
        # Stereo: transpose to (channels, samples)
        y_input = y.T
    
    # Apply time stretch with high-quality settings
    y2 = prb.time_stretch(
        y_input, 
        sr, 
        rate, 
        rbargs={
            "--formant": "",      # Preserve formants (empty string, not "on")
            "--pitch-hq": ""      # High-quality pitch shifting
        }
    )
    
    # Convert back to (samples, channels) if stereo
    if y.ndim == 1:
        y2_output = y2
    else:
        y2_output = y2.T
    
    # Fix length exactly by padding or trimming
    current_length = len(y2_output) if y2_output.ndim == 1 else y2_output.shape[0]
    
    if current_length < target_samples:
        # Pad with silence
        pad_samples = target_samples - current_length
        if y2_output.ndim == 1:
            pad = np.zeros(pad_samples, dtype=y2_output.dtype)
            y2_output = np.concatenate([y2_output, pad])
        else:
            pad = np.zeros((pad_samples, y2_output.shape[1]), dtype=y2_output.dtype)
            y2_output = np.vstack([y2_output, pad])
        print(f"   Padded {pad_samples} samples")
    elif current_length > target_samples:
        # Trim excess
        if y2_output.ndim == 1:
            y2_output = y2_output[:target_samples]
        else:
            y2_output = y2_output[:target_samples, :]
        print(f"   Trimmed {current_length - target_samples} samples")
    
    # Write output
    sf.write(out_wav, y2_output, sr)
    print(f"   ✅ Saved: {out_wav}")
    
    return out_wav




def adjust_audio_speed(input_files: List[Dict], output_dir: Optional[Path] = None) -> List[Dict]:
    """
    Adjust speed of multiple audio segments to match their target durations.
    
    Args:
        input_files: List of dicts with 'audio_url', 'start', 'end', 'speaker_id'
        output_dir: Output directory for processed files (default: ./outs)
    
    Returns:
        list: List of dicts with 'path', 'start', 'end', 'speaker_id'
    """
    if output_dir is None:
        output_dir = Path("./outs")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎵 Processing {len(input_files)} audio segments...")
    
    resized = []
    
    for i, tts_seg in enumerate(input_files):
        audio_url = tts_seg.get("audio_url")
        
        if not audio_url:
            print(f"⚠️  Segment {i}: Missing 'audio_url' field")
            continue
        
        audio_path = Path(audio_url)
        
        if not audio_path.exists():
            print(f"⚠️  Segment {i}: File not found: {audio_path}")
            continue
        
        # Calculate target duration from start/end times
        target_duration = tts_seg["end"] - tts_seg["start"]
        
        # Generate output filename
        output_file = output_dir / f"resized_seg_{i}.wav"
        
        try:
            # Use the granular function to process this segment
            result = rubberband_to_duration( 
                in_wav=audio_path,
                target_ms=target_duration*1000,
                out_wav=output_file
            )
            
            # Add to results
            resized.append({
                "path": str(result),
                "start": tts_seg["start"],
                "end": tts_seg["end"],
                "speaker_id": tts_seg.get("speaker_id")
            })
            
        except Exception as e:
            print(f"❌ Error processing segment {i}: {e}")
            continue
    
    print(f"\n✅ Successfully processed {len(resized)}/{len(input_files)} segments")
    
    return resized


def link_audio_segments_simple(audio_files, output_file, crossfade_duration=0.5):
    """
    Simplified version: just link audio files with crossfade (no timing adjustment).
    
    Args:
        audio_files (list): List of paths to audio files to be linked
        output_file (str): Path for the output linked audio file
        crossfade_duration (float): Duration of crossfade between segments in seconds (default: 0.5)
    
    Returns:
        str: Path to the linked audio file
    """
    if len(audio_files) < 2:
        raise ValueError("At least 2 audio files are required for linking")
    
    # Build ffmpeg command with crossfade filters
    cmd = ['ffmpeg', '-y']  # -y to overwrite output file
    
    # Add input files
    for audio_file in audio_files:
        cmd.extend(['-i', audio_file])
    
    # Build filter complex for crossfading
    filter_parts = []
    
    # For each pair of adjacent files, create a crossfade
    for i in range(len(audio_files) - 1):
        if i == 0:
            # First crossfade with triangular curve for smooth transition
            filter_parts.append(f"[0][1]acrossfade=d={crossfade_duration}:c1=tri:c2=tri[cf{i}]")
        else:
            # Subsequent crossfades
            filter_parts.append(f"[cf{i-1}][{i+1}]acrossfade=d={crossfade_duration}:c1=tri:c2=tri[cf{i}]")
    
    # Combine all filter parts
    filter_complex = ";".join(filter_parts)
    
    # Add filter complex to command
    cmd.extend(['-filter_complex', filter_complex])
    
    # Map the final output
    final_output = f"cf{len(audio_files) - 2}" if len(audio_files) > 2 else "cf0"
    cmd.extend(['-map', f'[{final_output}]'])
    
    # Add output file
    cmd.append(output_file)
    
    print(f"Linking {len(audio_files)} audio segments with {crossfade_duration}s crossfade...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Execute ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully linked audio segments to: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error linking audio segments: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        raise


def concatenate_audio_simple(audio_files, output_file):
    """
    Simple concatenation of audio files without crossfade using ffmpeg.
    
    Args:
        audio_files (list): List of paths to audio files to be concatenated
        output_file (str): Path for the output concatenated audio file
    
    Returns:
        str: Path to the concatenated audio file
    """
    if len(audio_files) < 2:
        raise ValueError("At least 2 audio files are required for concatenation")
    
    # Create temporary file list for ffmpeg concat demuxer
    import tempfile
    temp_dir = tempfile.mkdtemp()
    filelist_path = os.path.join(temp_dir, "temp_filelist.txt")
    
    try:
        # Write file list
        with open(filelist_path, 'w') as f:
            for audio_file in audio_files:
                f.write(f"file '{os.path.abspath(audio_file)}'\n")
        
        # Build ffmpeg command for concatenation
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',
            output_file
        ]
        
        print(f"Concatenating {len(audio_files)} audio files...")
        
        # Execute ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully concatenated audio files to: {output_file}")
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error concatenating audio files: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        raise
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)




def overlay_on_background_librosa(dubbed_segments, background_path: Path, output_path: Path, 
                                   original_duration: float = None, background_volume=0.12):
    """
    Overlay dubbed segments on background with intelligent ducking and smart timing.
    - Each segment starts at its scheduled time OR right after the previous one ends
    - No clipping: extends to last segment
    - Dynamic ducking with smooth transitions
    - Voice-over always louder than background
    
    Args:
        dubbed_segments: List of dicts with 'path', 'start', 'end', 'speaker_id'
        background_path: Path to background audio
        output_path: Path for output file
        original_duration: Optional original duration (will extend if needed)
        background_volume: Background volume during speech (0.05-0.25, default 0.12)
    """
    print(f"🎵 Processing {len(dubbed_segments)} segments with librosa...")
    
    if not dubbed_segments:
        print("⚠️  No segments to process!")
        return None
    
    # Step 1: Calculate actual timing for each segment
    print("📐 Calculating segment timing...")
    adjusted_segments = []
    previous_end_time = 0.0
    
    for i, seg in enumerate(dubbed_segments):
        scheduled_start = seg["start"]
        scheduled_end = seg["end"]
        
        # Load segment to get actual duration
        seg_audio, seg_sr = librosa.load(seg["path"], sr=44100, mono=False)
        if seg_audio.ndim == 1:
            actual_duration = len(seg_audio) / seg_sr
        else:
            actual_duration = seg_audio.shape[1] / seg_sr
        
        # Determine actual start time
        if i == 0:
            # First segment always starts at scheduled time
            actual_start = scheduled_start
        else:
            # Start at scheduled time OR right after previous segment (whichever is later)
            actual_start = max(scheduled_start, previous_end_time)
        
        actual_end = actual_start + actual_duration
        
        adjusted_segments.append({
            "path": seg["path"],
            "scheduled_start": scheduled_start,
            "scheduled_end": scheduled_end,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "actual_duration": actual_duration,
            "speaker_id": seg.get("speaker_id")
        })
        
        previous_end_time = actual_end
        
        # Log timing info
        if actual_start != scheduled_start:
            print(f"  Segment {i}: Scheduled {scheduled_start:.2f}s → Delayed to {actual_start:.2f}s (previous segment overlap)")
        else:
            print(f"  Segment {i}: Starts at scheduled time {actual_start:.2f}s")
    
    # Calculate required duration (extend to last segment)
    last_segment_end = adjusted_segments[-1]["actual_end"]
    required_duration = max(original_duration or 0, last_segment_end + 0.5)
    
    print(f"📏 Required duration: {required_duration:.2f}s (last segment ends: {last_segment_end:.2f}s)")
    
    # Step 2: Load and prepare background audio
    background, sr = librosa.load(str(background_path), sr=44100, mono=False)
    if background.ndim == 1:
        background = np.stack([background, background])
    
    print(f"🎼 Background loaded: {background.shape[1]/sr:.2f}s @ {sr}Hz")
    
    # Calculate target samples
    target_samples = int(required_duration * sr)
    
    # Extend/loop background if needed
    if background.shape[1] < target_samples:
        times_to_loop = int(np.ceil(target_samples / background.shape[1]))
        background = np.tile(background, (1, times_to_loop))
        print(f"🔁 Looped background {times_to_loop} times")
    
    # Trim to exact length
    background = background[:, :target_samples]
    
    # Step 3: Create voice track with adjusted timing
    voice_track = np.zeros((2, target_samples))
    
    print("🎤 Overlaying voice segments with adjusted timing...")
    for i, seg in enumerate(adjusted_segments):
        seg_audio, seg_sr = librosa.load(seg["path"], sr=sr, mono=False)
        
        if seg_audio.ndim == 1:
            seg_audio = np.stack([seg_audio, seg_audio])
        
        # Use ACTUAL start time (not scheduled)
        start_sample = int(seg["actual_start"] * sr)
        seg_length = seg_audio.shape[1]
        end_sample = min(start_sample + seg_length, target_samples)
        actual_length = end_sample - start_sample
        
        if actual_length > 0:
            # Place segment at actual start time
            voice_track[:, start_sample:end_sample] = seg_audio[:, :actual_length]
            print(f"  ✓ Segment {i}: {seg['actual_start']:.2f}s - {seg['actual_end']:.2f}s "
                  f"(duration: {seg['actual_duration']:.2f}s)")
    
    # Step 4: Create dynamic ducking mask based on ACTUAL timing
    print("📉 Creating intelligent ducking mask...")
    ducking_envelope = np.ones(target_samples)  # Start at full volume
    
    for seg in adjusted_segments:
        start_sample = int(seg["actual_start"] * sr)
        end_sample = int(min(seg["actual_end"], required_duration) * sr)
        
        # Attack/release times
        attack_samples = int(0.1 * sr)  # 100ms fade down
        release_samples = int(0.3 * sr)  # 300ms fade up
        
        # Calculate attack region (fade down before speech)
        attack_start = max(0, start_sample - attack_samples)
        attack_region = np.linspace(1.0, background_volume, attack_samples)
        attack_end = min(start_sample, target_samples)
        attack_len = attack_end - attack_start
        if attack_len > 0:
            ducking_envelope[attack_start:attack_end] = np.minimum(
                ducking_envelope[attack_start:attack_end],
                attack_region[:attack_len]
            )
        
        # Hold region (low volume during speech)
        hold_end = min(end_sample, target_samples)
        if start_sample < hold_end:
            ducking_envelope[start_sample:hold_end] = np.minimum(
                ducking_envelope[start_sample:hold_end],
                background_volume
            )
        
        # Release region (fade up after speech)
        release_start = end_sample
        release_end = min(end_sample + release_samples, target_samples)
        release_region = np.linspace(background_volume, 1.0, release_samples)
        release_len = release_end - release_start
        if release_len > 0:
            ducking_envelope[release_start:release_end] = np.minimum(
                ducking_envelope[release_start:release_end],
                release_region[:release_len]
            )
    
    # Smooth the envelope to avoid clicks
    window_size = int(0.01 * sr)  # 10ms smoothing
    if window_size % 2 == 0:
        window_size += 1
    ducking_envelope = signal.savgol_filter(ducking_envelope, window_size, 3)
    
    # Apply ducking to background
    background_ducked = background * ducking_envelope
    
    # Normalize voice track to prevent clipping
    voice_max = np.max(np.abs(voice_track))
    if voice_max > 0:
        voice_track = voice_track / voice_max * 0.9  # Leave headroom
        print(f"🎚️  Voice normalized (peak: {voice_max:.3f})")
    
    # Mix: voice (foreground) + ducked background
    final_mix = voice_track * 1.2 + background_ducked  # Boost voice slightly
    
    # Master limiting (prevent clipping)
    mix_max = np.max(np.abs(final_mix))
    if mix_max > 1.0:
        final_mix = final_mix / mix_max * 0.98
        print(f"🎚️  Limited mix (peak was: {mix_max:.3f})")
    
    # Apply gentle compression for consistent loudness
    final_mix = np.tanh(final_mix * 1.1) * 0.95
    
    # Save output
    sf.write(str(output_path), final_mix.T, sr, subtype='PCM_16')
    print(f"✅ Final audio saved: {output_path}")
    print(f"📊 Duration: {final_mix.shape[1]/sr:.2f}s, Sample rate: {sr}Hz")
    
    # Return adjusted segments info for subtitle timing correction
    return {
        "output_path": str(output_path),
        "adjusted_segments": adjusted_segments
    }


def overlay_on_background(dubbed_segments, background_path: Path, output_path: Path, 
                          original_duration: float = None, background_volume=0.12):
    """
    Overlay with intelligent ducking and smart timing using librosa.
    Returns adjusted segment timing information.
    """
    return overlay_on_background_librosa(
        dubbed_segments, background_path, output_path,
        original_duration, background_volume
    )


def get_audio_duration(audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())