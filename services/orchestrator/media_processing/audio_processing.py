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

def check_audio_structure(audio_file: str):
    import torchaudio
    import torch

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

    import matplotlib.pyplot as plt

    signal, fs = torchaudio.load(audio_file)
    signal = signal.squeeze()
    time = torch.linspace(0, signal.shape[0]/fs, steps=signal.shape[0])

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


def concatenate_audio(segments, output_file, target_duration: Optional[float] = None, min_silence_gap=0.02, max_silence_gap=1.0):
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
    audio_files = [seg.audio_url for seg in segments]
    
    # ========== SINGLE SEGMENT CASE ==========
    if len(segments) == 1:
        if target_duration is None:
            shutil.copy2(audio_files[0], output_file)
            print(f"Single file copied to: {output_file}")
            return output_file
        else:
            print(f"Adjusting single file to target duration: {target_duration:.2f}s")
            return rubberband_to_duration(audio_files[0], target_duration * 1000, output_file)
    
    # ========== NO TARGET DURATION ==========
    if target_duration is None:
        return _simple_concat(audio_files, output_file)
    
    # ========== ANALYZE SEGMENTS (single pass) ==========
    print("\n" + "="*60)
    print("📊 ANALYZING SEGMENTS")
    print("="*60)
    
    segment_info = []
    total_actual_duration = 0.0
    total_expected_duration = 0.0
    
    for i, seg in enumerate(segments):
        expected_duration = seg.end - seg.start
        actual_duration = get_audio_duration(Path(seg.audio_url))
        is_longer = actual_duration > expected_duration
        
        segment_info.append({
            'index': i,
            'audio_url': seg.audio_url,
            'start': seg.start,
            'end': seg.end,
            'expected_duration': expected_duration,
            'actual_duration': actual_duration,
            'is_longer': is_longer,
        })
        
        total_actual_duration += actual_duration
        total_expected_duration += expected_duration
        
        status = "🔴 LONGER" if is_longer else "🟢 OK"
        print(f"Segment {i}: {status} | Expected: {expected_duration:.2f}s | Actual: {actual_duration:.2f}s")
    
    print(f"\nTotal actual: {total_actual_duration:.2f}s | Expected: {total_expected_duration:.2f}s | Target: {target_duration:.2f}s")
    
    # ========== CASE 1: Total actual > target ==========
    # Need to compress everything to fit
    if total_actual_duration > target_duration:
        print(f"\n⚠️  Audio ({total_actual_duration:.2f}s) exceeds target ({target_duration:.2f}s)")
        print(f"   Compressing by {(total_actual_duration/target_duration):.3f}x")
        
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        try:
            temp_concat = temp_dir / "temp_concat.wav"
            _simple_concat(audio_files, str(temp_concat))
            result = rubberband_to_duration(str(temp_concat), target_duration * 1000, output_file)
            print("✅ COMPRESSION COMPLETE\n" + "="*60)
            return result
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # ========== CASE 2: Total actual >= total expected ==========
    # No stretching needed, just add silence to reach target
    if total_actual_duration >= total_expected_duration:
        print(f"\n✅ Sum of all segments durations above expected sum of durations but below target")
        print(f"   Adding silence to reach target: {target_duration - total_actual_duration:.2f}s")
        
        processed_files = audio_files
        total_processed_duration = total_actual_duration
    
    # ========== CASE 3: Total actual < total expected ==========
    # Some segments need stretching
    else:
        print(f"\n🔧 Sum of all segments durations below expected sum of durations - applying intelligent stretching")
        
        # Separate longer vs normal segments
        longer_segments = [s for s in segment_info if s['is_longer']]
        normal_segments = [s for s in segment_info if not s['is_longer']]
        
        # Calculate adjustment ratio for normal segments
        if longer_segments:
            duration_used_by_longer = sum(s['actual_duration'] for s in longer_segments)
            expected_duration_of_longer = sum(s['expected_duration'] for s in longer_segments)
            
            remaining_target = target_duration - duration_used_by_longer
            hypothetical_remaining = target_duration - expected_duration_of_longer
            
            adjustment_ratio = remaining_target / hypothetical_remaining if hypothetical_remaining > 0 else 1.0
            
            print(f"   Longer segments: {len(longer_segments)} (using {duration_used_by_longer:.2f}s)")
            print(f"   Normal segments: {len(normal_segments)} (ratio: {adjustment_ratio:.3f}x)")
        else:
            # All segments below expected - scale proportionally
            adjustment_ratio = 0.0
            print(f"   All segments below expected - scaling by {adjustment_ratio:.3f}x")
        
        # Process segments
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        processed_files = []
        total_processed_duration = 0.0
        
        try:
            for seg_info in segment_info:
                if seg_info['is_longer']:
                    # Keep as-is
                    processed_files.append(seg_info['audio_url'])
                    seg_info['final_duration'] = seg_info['actual_duration']
                    print(f"  ✓ Segment {seg_info['index']}: Keep original ({seg_info['actual_duration']:.2f}s)")
                else:
                    # Calculate new expected duration
                    new_expected = seg_info['expected_duration'] * adjustment_ratio
                    
                    if seg_info['actual_duration'] < new_expected:
                        # Needs stretching
                        stretched_file = temp_dir / f"stretched_seg_{seg_info['index']}.wav"
                        rubberband_to_duration(
                            seg_info['audio_url'],
                            new_expected * 1000,
                            str(stretched_file)
                        )
                        processed_files.append(str(stretched_file))
                        seg_info['final_duration'] = new_expected
                        print(f"  🔄 Segment {seg_info['index']}: Stretch {seg_info['actual_duration']:.2f}s → {new_expected:.2f}s")
                    else:
                        # Actual is already sufficient
                        processed_files.append(seg_info['audio_url'])
                        seg_info['final_duration'] = seg_info['actual_duration']
                        print(f"  ✓ Segment {seg_info['index']}: Keep original ({seg_info['actual_duration']:.2f}s)")
                
                total_processed_duration += seg_info['final_duration']
        
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
    # ========== ADD WEIGHTED SILENCE ==========
    silence_needed = target_duration - total_processed_duration
    
    if silence_needed > 0:  # Only if meaningful silence is needed
        print(f"\n🔇 Adding {silence_needed:.2f}s weighted silence")
        
        # Calculate original gaps
        original_gaps = [segments[0].start]  # Before first
        for i in range(len(segments) - 1):
            original_gaps.append(segments[i + 1].start - segments[i].end)
        original_gaps.append(target_duration - segments[-1].end)  # After last
        
        total_original_gaps = sum(original_gaps)
        
        # Weight silence proportionally
        if total_original_gaps > 0:
            silence_gaps = [(gap / total_original_gaps) * silence_needed for gap in original_gaps]
        else:
            silence_gaps = [silence_needed / len(original_gaps)] * len(original_gaps)
        
        print(f"   Original gaps: {[f'{g:.2f}' for g in original_gaps]}")
        print(f"   Silence gaps: {[f'{s:.2f}' for s in silence_gaps]}")
        
        result = _concat_with_weighted_silence(processed_files, output_file, silence_gaps)
    
    else:
        # Processed duration exceeds target: compress final concat to exact target
        over_by = -silence_needed
        print(f"\n⚠️  Processed duration exceeds target by {over_by:.3f}s → compressing to target")
        import tempfile
        temp_dir2 = Path(tempfile.mkdtemp())
        try:
            temp_concat = temp_dir2 / "temp_concat.wav"
            _simple_concat(processed_files, str(temp_concat))
            result = rubberband_to_duration(str(temp_concat), target_duration * 1000, output_file)
        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)
    
    # Cleanup temp directory if it exists
    if 'temp_dir' in locals():
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "="*60)
    print("✅ CONCATENATION COMPLETE")
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