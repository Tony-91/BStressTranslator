"""
Extract log-mel spectrogram features from audio slices.

Input: dataset_clean_slices/
Output: dataset_features/logmel/
"""
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLICES_DIR = PROJECT_ROOT / "dataset_clean_slices"
FEATURES_DIR = PROJECT_ROOT / "dataset_features" / "logmel"

TARGET_SR = 16000
N_MELS = 64            # Number of mel bands
HOP_LENGTH = 256       # Frames hop length
N_FFT = 512            # FFT window size

# --- Helper function ---
def extract_logmel(y, sr):
    """Convert audio waveform to log-mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.astype(np.float32)

# --- Main pipeline ---
def run_pipeline(slices_dir: Path, features_dir: Path):
    for class_dir in slices_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        feature_class_dir = features_dir / class_name
        feature_class_dir.mkdir(parents=True, exist_ok=True)

        slice_files = list(class_dir.glob("*.wav"))
        for slice_file in tqdm(slice_files, desc=f"Processing {class_name}"):
            try:
                y, sr = librosa.load(slice_file, sr=TARGET_SR, mono=True)
                logmel = extract_logmel(y, sr)
                # Save as numpy file
                out_file = feature_class_dir / f"{slice_file.stem}.npy"
                np.save(out_file, logmel)
            except Exception as e:
                print(f"Error processing {slice_file.name}: {e}")

    print(f"\nFeature extraction complete! Features saved in {features_dir}")

if __name__ == "__main__":
    run_pipeline(SLICES_DIR, FEATURES_DIR)
