import csv
from pathlib import Path
from typing import Dict, Any
from generate_metadata import find_audio_files
import soundfile as sf
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLICES_DIR = PROJECT_ROOT / "dataset_clean_slices"
OUTPUT_CSV = SLICES_DIR / "metadata" / "slice_metadata.csv"


def get_slice_metadata(audio_path: Path) -> Dict[str, Any]:
    """Metadata extractor specialized for sliced audio."""
    file_size = os.path.getsize(audio_path)

    with sf.SoundFile(str(audio_path)) as f:
        duration = f.frames / f.samplerate

    stem_parts = audio_path.stem.split('_')

    return {
        "filename": audio_path.name,
        "file_path": str(audio_path),
        "class": audio_path.parent.name,
        "sample_rate": f.samplerate,
        "duration_seconds": round(duration, 2),
        "channels": f.channels,
        "file_size_mb": round(file_size / (1024 * 1024), 3),
        "slice_id": next((p for p in stem_parts if p.startswith("w")), "unknown")
    }


def generate_slice_metadata():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(SLICES_DIR)
    metadata = []

    for audio_file in audio_files:
        try:
            metadata.append(get_slice_metadata(audio_file))
        except Exception as e:
            print(f"Skipping {audio_file.name}: {e}")

    fieldnames = [
        "filename",
        "file_path",
        "class",
        "sample_rate",
        "duration_seconds",
        "channels",
        "file_size_mb",
        "slice_id",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Generated slice metadata for {len(metadata)} files")


if __name__ == "__main__":
    generate_slice_metadata()
