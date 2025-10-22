import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import librosa
import soundfile as sf
import numpy as np
import subprocess
import json
from scipy import signal
import pyrubberband as prb
import os
import shutil
import torch
import torchaudio
import tempfile


def check_audio_structure(audio_file: str):

    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    # Read audio file
    wav = read_audio(audio_file)
    
    # Get speech timestamps (returns sample indices by default)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True  # ← Add this parameter to get seconds instead of samples
    )

    # Extract speech timestamps
    for i, segment in enumerate(speech_timestamps):
        print(f"Speech segment {i}: {segment['start']:.2f}s - {segment['end']:.2f}s")

    # Get the last speech timestamp
    if speech_timestamps:
        last_speech_end = speech_timestamps[-1]['end']
        print(f"\nLast speech ends at: {last_speech_end:.2f}s")
        
        # Trim audio to last speech end
        waveform, sr = torchaudio.load(audio_file)
        trim_sample = int(last_speech_end * sr) + int(0.1 * sr)  # Add 100ms padding
        trimmed_waveform = waveform[:, :trim_sample]
        
        # Save trimmed audio
        torchaudio.save('trimmed_output.wav', trimmed_waveform, sr)
        print(f"Saved trimmed audio (duration: {trim_sample/sr:.2f}s)")

    signal, fs = torchaudio.load(audio_file)
    signal = signal.squeeze()
    time = torch.linspace(0, signal.shape[0]/fs, steps=signal.shape[0])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.grid(True)
    
    # Mark speech regions
    if speech_timestamps:
        for seg in speech_timestamps:
            plt.axvspan(seg['start'], seg['end'], alpha=0.3, color='green')
    
    plt.show()
    from IPython.display import Audio

    return Audio(audio_file)


def trim_audio_with_vad(
    audio_path: str | Path,
    output_path: str | Path = "",
    sampling_rate: int = 16000,
    max_silence_gap_ms: int = 1000,
    several_seg: bool = False
) -> Union[ Tuple[float, str], Tuple[List[float], List[str]] ]:
    """
    Trim audio file at the first silence gap longer than max_silence_gap_ms.
    
    Args:
        audio_path: Input audio file path
        output_path: Output trimmed audio file path (or directory if several_seg=True)
        sampling_rate: Target sampling rate for VAD model
        max_silence_gap_ms: Maximum silence gap allowed between speech segments (default: 1000ms)
        several_seg: If True, save individual speech segments instead of one trimmed file
    
    Returns:
        If several_seg=False: Duration of trimmed audio in seconds
        If several_seg=True: List of durations for each saved segment
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    
    # Auto-add .wav extension if missing and saving single file
    if not several_seg and not output_path.suffix:
        output_path = output_path.with_suffix('.wav')
    
    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    # Read audio file
    wav = read_audio(str(audio_path), sampling_rate=sampling_rate)
    
    # Get speech timestamps with return_seconds=True to get time in seconds
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True  # Get timestamps in seconds instead of samples
    )
    
    if not speech_timestamps:
        # No speech detected, keep original file
        if audio_path != output_path:
            import shutil
            shutil.copy(audio_path, output_path)
        return 0.0 if not several_seg else []
    
    # Load original audio for trimming (preserve original sample rate)
    waveform, original_sr = torchaudio.load(str(audio_path))
    
    # Find first silence gap longer than max_silence_gap_ms
    max_silence_gap_sec = max_silence_gap_ms / 1000.0
    cut_index = None
    
    for i in range(len(speech_timestamps) - 1):
        current_end = speech_timestamps[i]['end']
        next_start = speech_timestamps[i + 1]['start']
        silence_gap = next_start - current_end
        
        if silence_gap > max_silence_gap_sec:
            # Found a gap longer than threshold - cut here
            cut_index = i
            print(f"✂️  Found silence gap of {silence_gap:.2f}s between segments {i} and {i+1}")
            print(f"   Will use segments 0 to {i}")
            break
    
    # Determine which segments to use
    if cut_index is None:
        # No long gap found, use all segments
        segments_to_use = speech_timestamps
        print(f"✅ No silence gap > {max_silence_gap_sec:.2f}s found, using all {len(speech_timestamps)} segments")
    else:
        # Use segments up to and including the one before the gap
        segments_to_use = speech_timestamps[:cut_index + 1]
    
    if several_seg:
        # Save individual speech segments
        # Use output_path as directory (strip extension if present)
        output_dir = output_path.parent / output_path.stem if output_path.suffix else output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        durations = []
        output_files = []
        for i, seg in enumerate(segments_to_use):
            # Convert timestamps to samples
            start_sample = int(seg['start'] * original_sr)
            end_sample = int(seg['end'] * original_sr)
            
            # Add small padding (e.g., 50ms on each side)
            padding_samples = int(0.05 * original_sr)
            start_sample = max(0, start_sample - padding_samples)
            end_sample = min(waveform.shape[1], end_sample + padding_samples)
            
            # Extract segment
            segment_waveform = waveform[:, start_sample:end_sample]

            # Get the base name of the audio file (without extension)
            audio_basename = audio_path.stem
            segment_file = output_dir / f"{audio_basename}_segment_{i:03d}.wav"
            # Save segment
            torchaudio.save(str(segment_file), segment_waveform, original_sr)
            output_files.append(str(segment_file))
            
            duration = segment_waveform.shape[1] / original_sr
            durations.append(duration)
            
            print(f"💾 Saved segment {i}: {segment_file}")
            print(f"   Duration: {duration:.2f}s ({seg['start']:.2f}s - {seg['end']:.2f}s)")
        
        print(f"\n✅ Saved {len(durations)} segments to {output_dir}")
        print(f"   Total duration: {sum(durations):.2f}s")
        
        return durations, output_files
    
    else:
        # Original behavior: save one trimmed file up to cut point
        cut_point = segments_to_use[-1]['end']
        
        # Convert cut point from seconds to samples
        trim_sample = int(cut_point * original_sr)
        
        # Add small padding (e.g., 100ms) to avoid cutting off speech
        padding_samples = int(0.1 * original_sr)
        trim_sample = min(trim_sample + padding_samples, waveform.shape[1])
        
        # Trim waveform
        trimmed_waveform = waveform[:, :trim_sample]
        
        # Save trimmed audio
        torchaudio.save(str(output_path), trimmed_waveform, original_sr)
        
        # Return duration in seconds
        duration = trimmed_waveform.shape[1] / original_sr
        
        print(f"💾 Saved trimmed audio: {output_path}")
        print(f"   Original duration: {waveform.shape[1] / original_sr:.2f}s")
        print(f"   Trimmed duration: {duration:.2f}s")
        print(f"   Removed: {(waveform.shape[1] / original_sr) - duration:.2f}s")

        return duration, str(output_path)

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
    # elif current_length > target_samples:
    #     # Trim excess
    #     if y2_output.ndim == 1:
    #         y2_output = y2_output[:target_samples]
    #     else:
    #         y2_output = y2_output[:target_samples, :]
    #     print(f"   Trimmed {current_length - target_samples} samples")
    
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


def concatenate_audio(segments, output_file, target_duration: Optional[float] = None):
    """
    Concatenate audio segments with intelligent duration adjustment.
    
    Args:
        segments (list): List of segment dicts with keys: 'audio_url', 'start', 'end'
        output_file (str): Path for the output concatenated audio file
        target_duration (float, optional): Target duration in seconds. If provided:
            - Segments longer than expected keep their actual duration
            - Remaining time is distributed proportionally to other segments
            - Segments are stretched if needed to fill their allocated time
            - Silence is added between segments to match target duration
    
    Returns:
        str: Path to the concatenated audio file
    """
    if len(segments) < 1:
        raise ValueError("At least 1 segment is required for concatenation")
    
    # Extract audio files
    audio_files = [seg["audio_url"] for seg in segments]
    
    # ========== SINGLE SEGMENT CASE ==========
    if len(segments) == 1:
        if target_duration:
            actual_duration = get_audio_duration(Path(audio_files[0]))
            if actual_duration > target_duration:
                # Compress to fit target duration
                rubberband_to_duration(audio_files[0], target_duration * 1000, output_file)
            else:
                # Pad with silence to reach target duration
                expected_duration = segments[0]["end"] - segments[0]["start"]
                
                # Calculate new timing to center the audio
                adjust_amount = expected_duration - actual_duration
                new_start = segments[0]["start"] + (adjust_amount / 2 if adjust_amount > 0 else 0)
                new_end = segments[0]["end"] - (adjust_amount / 2 if adjust_amount > 0 else 0)
                
                # Create silence gaps
                silence_before = new_start  # From 0 to new_start
                silence_after = target_duration - new_end  # From new_end to target_duration
                
                # Use weighted silence concatenation
                _concat_with_weighted_silence([audio_files[0]], output_file, [silence_before, silence_after])
        else:
            # No target duration, just copy the file
            shutil.copy(audio_files[0], output_file)
        
        return output_file
    
    # ========== NO TARGET DURATION ==========
    if target_duration is None:
        return _simple_concat(audio_files, output_file)
    
    # ========== CENTERING LOGIC WITH SILENCE PADDING ==========
    print("\n" + "="*60)
    print("📊 CENTERING SEGMENTS WITH SILENCE PADDING")
    print("="*60)
    
    segment_info = []
    
    # Calculate centered positions for each segment and track overrun amounts
    for i, seg in enumerate(segments):
        expected_duration = seg["end"] - seg["start"]
        actual_duration = get_audio_duration(Path(seg["audio_url"]))
        
        # Calculate overrun (how much longer than expected, 0 if shorter)
        overrun = max(0, actual_duration - expected_duration)
        
        # Center the actual audio within the expected time slot
        expected_center = (seg["start"] + seg["end"]) / 2
        centered_start = expected_center - (actual_duration / 2)
        centered_end = centered_start + actual_duration
        
        segment_info.append({
            'index': i,
            'audio_url': seg["audio_url"],
            'original_start': seg["start"],
            'original_end': seg["end"],
            'expected_duration': expected_duration,
            'actual_duration': actual_duration,
            'centered_start': centered_start,
            'centered_end': centered_end,
            'overrun': overrun,  # Track how much longer than expected
            'remaining_overrun': overrun,  # Track remaining overrun after adjustments
        })
        
        print(f"Segment {i}: Expected [{seg['start']:.2f}-{seg['end']:.2f}] → Centered [{centered_start:.2f}-{centered_end:.2f}] (overrun: {overrun:.3f}s)")
    
    # ========== HANDLE FIRST SEGMENT (shift to start at 0 if needed) ==========
    if segment_info[0]['centered_start'] < 0:
        shift = -segment_info[0]['centered_start']
        segment_info[0]['centered_start'] = 0
        segment_info[0]['centered_end'] += shift
        print(f"🔧 First segment shifted by +{shift:.3f}s to start at 0")
    
    # ========== HANDLE LAST SEGMENT (shift to end at target_duration if needed) ==========
    if segment_info[-1]['centered_end'] > target_duration:
        shift = segment_info[-1]['centered_end'] - target_duration
        segment_info[-1]['centered_end'] = target_duration
        segment_info[-1]['centered_start'] -= shift
        print(f"🔧 Last segment shifted by -{shift:.3f}s to end at target duration")
    
    # ========== RESOLVE OVERLAPS WITH WEIGHTED ADJUSTMENT ==========
    print("\n🔍 Checking for overlaps...")
    for i in range(len(segment_info) - 1):
        current = segment_info[i]
        next_seg = segment_info[i + 1]
        
        if current['centered_end'] > next_seg['centered_start']:
            overlap = current['centered_end'] - next_seg['centered_start']
            print(f"⚠️  Overlap detected between segments {i} and {i+1}: {overlap:.3f}s")
            
            # Calculate weights based on remaining overrun amounts
            current_weight = current['remaining_overrun']
            next_weight = next_seg['remaining_overrun']
            total_weight = current_weight + next_weight
            
            print(f"   Current segment overrun: {current_weight:.3f}s, Next segment overrun: {next_weight:.3f}s")
            
            if total_weight > 0:
                # Weighted adjustment based on how much each segment exceeded its expected duration
                current_adjustment = overlap * (current_weight / total_weight)
                next_adjustment = overlap * (next_weight / total_weight)
            else:
                # Neither segment has overrun, split equally
                current_adjustment = overlap / 2
                next_adjustment = overlap / 2
            
           
            # Normal case: weighted adjustment
            current['centered_end'] -= current_adjustment
            next_seg['centered_start'] += next_adjustment
            print(f"   → Adjusted segment {i} by -{current_adjustment:.3f}s, segment {i+1} by +{next_adjustment:.3f}s")
            
            # Update remaining overrun amounts after adjustment
            current['remaining_overrun'] = max(0, current['remaining_overrun'] - current_adjustment)
            next_seg['remaining_overrun'] = max(0, next_seg['remaining_overrun'] - next_adjustment)
            
    
    # ========== BUILD TIMELINE WITH SILENCE ==========
    print("\n🔇 Building timeline with silence padding...")
    
    # Create timeline events (segments + silence gaps)
    timeline = []
    current_time = 0.0
    
    for i, seg_info in enumerate(segment_info):
        # Add silence before segment if needed
        if current_time < seg_info['centered_start']:
            silence_duration = seg_info['centered_start'] - current_time
            timeline.append({
                'type': 'silence',
                'duration': silence_duration,
                'start': current_time,
                'end': seg_info['centered_start']
            })
            print(f"   Silence gap {len([t for t in timeline if t['type'] == 'silence'])}: {silence_duration:.3f}s")
        
        # Add segment
        timeline.append({
            'type': 'audio',
            'audio_url': seg_info['audio_url'],
            'duration': seg_info['actual_duration'],
            'start': seg_info['centered_start'],
            'end': seg_info['centered_end']
        })
        print(f"   Segment {i}: {seg_info['actual_duration']:.3f}s at [{seg_info['centered_start']:.3f}-{seg_info['centered_end']:.3f}]")
        
        current_time = seg_info['centered_end']
    
    # Add final silence if needed
    if current_time < target_duration:
        final_silence = target_duration - current_time
        timeline.append({
            'type': 'silence',
            'duration': final_silence,
            'start': current_time,
            'end': target_duration
        })
        print(f"   Final silence: {final_silence:.3f}s")
    
    # ========== CONCATENATE WITH FFMPEG ==========
    print(f"\n🎵 Concatenating {len(timeline)} elements...")
    
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Get audio properties from first audio segment
        first_audio = next(t for t in timeline if t['type'] == 'audio')['audio_url']
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,channels',
            '-of', 'json',
            first_audio
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        audio_info = json.loads(probe_result.stdout)
        sample_rate = audio_info['streams'][0]['sample_rate']
        channels = audio_info['streams'][0]['channels']
        
        # Create silence files and build file list
        filelist_path = temp_dir / "filelist.txt"
        silence_counter = 0
        
        with open(filelist_path, 'w') as f:
            for element in timeline:
                if element['type'] == 'silence':
                    if element['duration'] > 0.001:  # Only create if > 1ms
                        silence_file = temp_dir / f"silence_{silence_counter}.wav"
                        silence_cmd = [
                            'ffmpeg', '-y',
                            '-f', 'lavfi',
                            '-i', f'anullsrc=channel_layout={"stereo" if channels == 2 else "mono"}:sample_rate={sample_rate}',
                            '-t', str(element['duration']),
                            str(silence_file)
                        ]
                        subprocess.run(silence_cmd, capture_output=True, text=True, check=True)
                        f.write(f"file '{silence_file.absolute()}'\n")
                        silence_counter += 1
                else:
                    # Audio segment
                    f.write(f"file '{Path(element['audio_url']).absolute()}'\n")
        
        # Final concatenation
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(filelist_path),
            '-c', 'copy',
            str(output_file)
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Verify final duration
        final_duration = get_audio_duration(output_file)
        print(f"✅ Final duration: {final_duration:.3f}s (target: {target_duration:.3f}s)")
        
        if final_duration - target_duration > 0.0:
            print(f"⚠️  Duration mismatch > 0.1s, applying final trim/pad...")
            result = rubberband_to_duration(str(output_file), target_duration * 1000, str(output_file))
        else:
            result = str(output_file)
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "="*60)
    print("✅ CENTERING AND PADDING COMPLETE")
    print("="*60)
    
    return result


def _simple_concat(audio_files, output_file):
    """Helper: Simple concatenation without any adjustments"""
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
        shutil.rmtree(temp_dir, ignore_errors=True)


def _concat_with_weighted_silence(audio_files, output_file, silence_gaps):
    """
    Helper: Concatenate audio files with weighted silence gaps.
    
    Args:
        audio_files: List of audio file paths
        output_file: Output file path
        silence_gaps: List of silence durations. Length should be len(audio_files) + 1
                     [before_first, between_1_2, between_2_3, ..., after_last]
    """
    import tempfile
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Get sample rate and channels from first audio file
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,channels',
            '-of', 'json',
            audio_files[0]
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        audio_info = json.loads(probe_result.stdout)
        sample_rate = audio_info['streams'][0]['sample_rate']
        channels = audio_info['streams'][0]['channels']
        
        # Create silence files for each gap
        silence_files = []
        for i, silence_duration in enumerate(silence_gaps):
            if silence_duration > 0.001:  # Only create if > 1ms
                silence_file = os.path.join(temp_dir, f"silence_{i}.wav")
                silence_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi',
                    '-i', f'anullsrc=channel_layout={"stereo" if channels == 2 else "mono"}:sample_rate={sample_rate}',
                    '-t', str(silence_duration),
                    silence_file
                ]
                subprocess.run(silence_cmd, capture_output=True, text=True, check=True)
                silence_files.append(silence_file)
            else:
                silence_files.append(None)
        
        # Build file list with weighted silence
        filelist_path = os.path.join(temp_dir, "filelist.txt")
        with open(filelist_path, 'w') as f:
            # Silence before first segment
            if silence_files[0]:
                f.write(f"file '{os.path.abspath(silence_files[0])}'\n")
            
            # Interleave audio files and silence gaps
            for i, audio_file in enumerate(audio_files):
                f.write(f"file '{os.path.abspath(audio_file)}'\n")
                
                # Add silence after this segment (if not last segment or if silence after last exists)
                silence_idx = i + 1
                if silence_idx < len(silence_files) and silence_files[silence_idx]:
                    f.write(f"file '{os.path.abspath(silence_files[silence_idx])}'\n")
        
        # Concatenate with weighted silence
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',
            output_file
        ]
        
        print(f"Concatenating {len(audio_files)} files with weighted silence...")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Successfully concatenated with weighted silence to: {output_file}")
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error concatenating with weighted silence: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        raise
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def overlay_on_background_default(
    dubbed_segments: List[Dict],
    background_path: Path | str,
    output_path: Path | str,
    ducking_db: float = 0.0,
) -> str:
    """
    Overlay processed dubbed segments onto the background track.

    Args:
        dubbed_segments: Sequence of dicts with at least start, end, audio_url (seconds + file path).
        background_path: Path to the separated instrumental/background audio.
        output_path: Destination WAV path for the mixed result.
        original_duration: Optional clamp/pad duration in seconds.
        ducking_db: Gain reduction (dB) applied to the background during overlay (default 0).

    Returns:
        Path to the rendered mix as string.
    """
    if not dubbed_segments:
        raise ValueError("dubbed_segments must not be empty")

    background_path = Path(background_path)
    output_path = Path(output_path)

    if not background_path.exists():
        raise FileNotFoundError(f"Background track not found: {background_path}")

    bg_wave, sr = sf.read(str(background_path), always_2d=True)
    bg_wave = bg_wave.astype(np.float32)

    if ducking_db != 0.0:
        gain = 10 ** (ducking_db / 20.0)
        bg_wave *= gain

    mix = bg_wave.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for idx, seg in enumerate(dubbed_segments):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            audio_url = seg.get("audio_url")

            if audio_url is None:
                continue

            if end <= start:
                continue

            source_path = Path(audio_url)
            if not source_path.exists():
                continue

            target_ms = (end - start) * 1000.0
            adjusted_path = source_path
            source_duration = get_audio_duration(str(source_path))

            if abs(source_duration - (end - start)) > 1e-3:
                adjusted_path = tmpdir / f"segment_{idx:03d}.wav"
                rubberband_to_duration(str(source_path), target_ms, str(adjusted_path))

            seg_wave, seg_sr = sf.read(str(adjusted_path), always_2d=True)
            seg_wave = seg_wave.astype(np.float32)

            if seg_sr != sr:
                seg_wave = librosa.resample(seg_wave.T, orig_sr=seg_sr, target_sr=sr).T

            if seg_wave.shape[1] != mix.shape[1]:
                if seg_wave.shape[1] == 1 and mix.shape[1] == 2:
                    seg_wave = np.tile(seg_wave, (1, 2))
                elif seg_wave.shape[1] == 2 and mix.shape[1] == 1:
                    seg_wave = seg_wave.mean(axis=1, keepdims=True)
                else:
                    raise ValueError(
                        f"Unsupported channel layout: segment {seg_wave.shape[1]} vs background {mix.shape[1]}"
                    )

            start_idx = int(round(start * sr))
            end_idx = start_idx + seg_wave.shape[0]

            if end_idx > mix.shape[0]:
                pad = np.zeros((end_idx - mix.shape[0], mix.shape[1]), dtype=np.float32)
                mix = np.vstack([mix, pad])

            mix[start_idx:end_idx, :seg_wave.shape[1]] += seg_wave

    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak * 1.01

    sf.write(str(output_path), mix, sr)
    return str(output_path)

def overlay_on_background_sophisticated(
    dubbed_segments: List[Dict],
    background_path: Path | str,
    output_path: Path | str,
    ducking_db: float = 0.0,
) -> str:
    """
    Create a single speech track using concatenate_audio logic (timing-aware, duration-controlled),
    then overlay it on top of the background with optional dynamic ducking.

    Args:
        dubbed_segments: List of dicts with at least 'audio_url', 'start', 'end' (seconds).
        background_path: Path to instrumental/background audio.
        output_path: Destination WAV path for the mixed result.
        ducking_db: Background gain (dB) under speech. Use negative values to reduce background during speech.

    Returns:
        Path to the rendered mix as string.
    """
    if not dubbed_segments:
        raise ValueError("dubbed_segments must not be empty")

    background_path = Path(background_path)
    output_path = Path(output_path)

    if not background_path.exists():
        raise FileNotFoundError(f"Background track not found: {background_path}")

    # Determine target duration from background
    target_duration = get_audio_duration(str(background_path))


    # Build the single speech track using the same logic as concatenate_audio
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        speech_track = tmpdir / "speech_concat.wav"
        concatenate_audio(dubbed_segments, str(speech_track), target_duration=target_duration)

        # Load background and speech
        bg_wave, bg_sr = sf.read(str(background_path), always_2d=True)
        sp_wave, sp_sr = sf.read(str(speech_track), always_2d=True)

        # Resample speech to background SR if needed
        if sp_sr != bg_sr:
            sp_wave = librosa.resample(sp_wave.T, orig_sr=sp_sr, target_sr=bg_sr).T

        # Match channel count (map to background layout)
        if sp_wave.shape[1] != bg_wave.shape[1]:
            if sp_wave.shape[1] == 1 and bg_wave.shape[1] == 2:
                sp_wave = np.tile(sp_wave, (1, 2))
            elif sp_wave.shape[1] == 2 and bg_wave.shape[1] == 1:
                sp_wave = sp_wave.mean(axis=1, keepdims=True)
            else:
                raise ValueError(
                    f"Unsupported channel layout: speech {sp_wave.shape[1]} vs background {bg_wave.shape[1]}"
                )

        # Pad/trim to exact background length (concatenate_audio targets this already, but guard anyway)
        bg_len = bg_wave.shape[0]
        sp_len = sp_wave.shape[0]
        if sp_len < bg_len:
            pad = np.zeros((bg_len - sp_len, sp_wave.shape[1]), dtype=sp_wave.dtype)
            sp_wave = np.vstack([sp_wave, pad])
        elif sp_len > bg_len:
            sp_wave = sp_wave[:bg_len, :]

        # Prepare dynamic ducking envelope from speech
        # ducking_db < 0 lowers background under speech proportionally to speech amplitude
        if ducking_db != 0.0:
            # Mono envelope from speech magnitude
            env = np.mean(np.abs(sp_wave), axis=1).astype(np.float32)

            # Smooth envelope with a short moving average (~20 ms)
            win = max(1, int(0.02 * bg_sr))
            if win > 1:
                kernel = np.ones(win, dtype=np.float32) / win
                env = np.convolve(env, kernel, mode="same")

            # Normalize to [0,1]
            max_env = float(env.max()) if env.size > 0 else 0.0
            if max_env > 1e-8:
                env = env / max_env
            else:
                env = np.zeros_like(env, dtype=np.float32)

            # Optional: longer release smoothing (~100 ms) to avoid pumping
            rel_win = max(1, int(0.10 * bg_sr))
            if rel_win > 1:
                kernel_rel = np.ones(rel_win, dtype=np.float32) / rel_win
                env = np.convolve(env, kernel_rel, mode="same")

            # Gain curve: 1.0 (no speech) to min_gain (full speech)
            # If ducking_db is negative, min_gain < 1 (reduce). If positive, min_gain > 1 (boost).
            min_gain = float(10.0 ** (ducking_db / 20.0))
            gain_curve = 1.0 + (min_gain - 1.0) * env  # shape [T]
            # Expand to channels
            gain_curve = gain_curve[:, None]
        else:
            gain_curve = 1.0

        # Mix: ducked background + speech
        bg_wave = bg_wave.astype(np.float32)
        sp_wave = sp_wave.astype(np.float32)
        mix = bg_wave * gain_curve + sp_wave

        # Normalize to prevent clipping
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 1.0:
            mix = mix / (peak * 1.01)

        sf.write(str(output_path), mix, bg_sr)

    return str(output_path)

# overlay functions selector
def overlay_on_background(dubbed_segments: List[Dict],
    background_path: Path | str,
    output_path: Path | str,
    ducking_db: float = 0.0,
    sophisticated: bool = False
) -> str:
    """
    Overlay dubbed segments onto background track using selected method.

    Args:
        dubbed_segments: List of dicts with at least 'audio_url', 'start', 'end' (seconds).
        background_path: Path to instrumental/background audio.
        output_path: Destination WAV path for the mixed result.
        ducking_db: Background gain (dB) under speech.
        sophisticated: If True, use sophisticated overlay method.

    Returns:
        Path to the rendered mix as string.
    """
    if sophisticated:
        return overlay_on_background_sophisticated(
            dubbed_segments, background_path, output_path, ducking_db
        )
    else:
        return overlay_on_background_default(
            dubbed_segments, background_path, output_path, ducking_db
        )

def get_audio_duration(path: Path | str) -> float:
    """Get audio duration in seconds with high precision."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True
    ).strip()
    return float(out)