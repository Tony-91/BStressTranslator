"""
Master preprocessing pipeline for BStressTranslator:
1. Load raw audio (.m4a, .wav, etc.)
2. Convert stereo → mono, resample to 16 kHz
3. Normalize amplitude
4. Optional: trim silence
5. Slice processed audio into 2-second overlapping windows
6. Save cleaned and sliced files in structured folders

16 kHz: Optimized for Nano 128-core Maxwell GPU and 512 MB GPU memory for spectrogram generation.
Normalization: Ensures spectrogram amplitude is consistent across classes — critical for model convergence.
Silence trimming: preserves background noise while removing excessive dead space.
Output: .m4a file becomes a cleaned .wav

"""

import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "dataset_raw"
CLEAN_DIR = PROJECT_ROOT / "dataset_clean"
SLICES_DIR = PROJECT_ROOT / "dataset_clean_slices"

TARGET_SR = 16000        # 16 kHz
TRIM_SILENCE = True
SILENCE_TOP_DB = 20       # dB threshold
WINDOW_LENGTH_SEC = 2.0
OVERLAP_SEC = 1.0

AUDIO_EXTS = ['*.m4a', '*.wav', '*.mp3', '*.flac']

# --- Helper functions ---

def preprocess_audio_file(audio_path: Path, output_dir: Path):
    """Preprocess a single audio file (mono, resample, normalize, trim)."""
    out_file = output_dir / f"{audio_path.stem}_clean.wav"
    if out_file.exists():
        print(f"Skipping {audio_path.name}, already processed.")
        return out_file
    
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample if needed
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        
        # Trim silence
        if TRIM_SILENCE:
            y, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_file, y, sr)
        print(f"Saved preprocessed {out_file.name}")
        return out_file
    except Exception as e:
        print(f"Error preprocessing {audio_path.name}: {e}")
        return None

def slice_audio_file(audio_path: Path, output_dir: Path):
    """Slice a preprocessed audio file into overlapping windows."""
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        total_samples = len(y)
        window_samples = int(WINDOW_LENGTH_SEC * sr)
        hop_samples = int((WINDOW_LENGTH_SEC - OVERLAP_SEC) * sr)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        slice_count = 0
        
        for start in range(0, total_samples - window_samples + 1, hop_samples):
            end = start + window_samples
            y_slice = y[start:end]
            slice_count += 1
            slice_file = output_dir / f"{audio_path.stem}_w{slice_count:02d}.wav"
            sf.write(slice_file, y_slice, sr)
        
        print(f"Sliced {audio_path.name} into {slice_count} windows")
    except Exception as e:
        print(f"Error slicing {audio_path.name}: {e}")

def find_audio_files(directory: Path):
    """Find all audio files recursively with supported extensions."""
    files = []
    for ext in AUDIO_EXTS:
        files.extend(directory.glob(f"**/{ext}"))
    return files

# --- Main pipeline ---
def run_pipeline(raw_dir: Path, clean_dir: Path, slices_dir: Path):
    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        clean_class_dir = clean_dir / class_dir.name
        slices_class_dir = slices_dir / class_dir.name
        
        audio_files = find_audio_files(class_dir)
        for audio_file in audio_files:
            # Step 3: Preprocess
            preprocessed_file = preprocess_audio_file(audio_file, clean_class_dir)
            if preprocessed_file:
                # Step 4: Slice
                slice_audio_file(preprocessed_file, slices_class_dir)

if __name__ == "__main__":
    run_pipeline(RAW_DIR, CLEAN_DIR, SLICES_DIR)
