"""
Script to generate metadata for log-mel spectrogram features (.npy) slices.
Extracts class, sample ID, task ID, slice ID, shape, and duration.
"""
import os
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# --- Helper functions ---

def get_feature_metadata(feature_path: Path) -> Dict[str, Any]:
    """Extract metadata from a log-mel .npy file."""
    try:
        # Parse filename components
        stem_parts = feature_path.stem.split('_')
        sample_id = stem_parts[-4] if len(stem_parts) >= 4 else 'unknown'  # s01
        task_id = stem_parts[-3] if len(stem_parts) >= 3 else 'unknown'    # t01
        slice_id = stem_parts[-1]                                          # w01

        # Load .npy to get shape
        data = np.load(feature_path)
        n_mels, n_frames = data.shape

        # Extract class from parent directory
        class_name = feature_path.parent.name

        return {
            'filename': feature_path.name,
            'file_path': str(feature_path.absolute()),
            'class': class_name,
            'sample_id': sample_id,
            'task_id': task_id,
            'slice_id': slice_id,
            'n_mels': n_mels,
            'n_frames': n_frames,
            'duration_seconds': 2.0  # all slices are 2 seconds
        }
    except Exception as e:
        print(f"Error processing {feature_path}: {e}")
        return None

def find_feature_files(directory: Path) -> List[Path]:
    """Find all log-mel .npy files in directory."""
    return list(directory.glob('**/*.npy'))

def generate_feature_metadata_csv(feature_dir: str, output_csv: str) -> None:
    """Generate metadata CSV for all log-mel feature files."""
    feature_dir = Path(feature_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_files = find_feature_files(feature_dir)
    if not feature_files:
        print(f"No .npy feature files found in {feature_dir}")
        return

    metadata = []
    for ffile in feature_files:
        meta = get_feature_metadata(ffile)
        if meta:
            metadata.append(meta)

    # Sort by class, sample_id, slice_id
    metadata.sort(key=lambda x: (x['class'], x['sample_id'], x['slice_id']))

    # Define CSV fields
    fieldnames = [
        'filename', 'file_path', 'class', 'sample_id', 'task_id',
        'slice_id', 'n_mels', 'n_frames', 'duration_seconds'
    ]

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Generated metadata for {len(metadata)} feature files at {output_path}")

# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description='Generate metadata for log-mel spectrogram slices')
    parser.add_argument('--input', '-i', default='../dataset_features/logmel',
                        help='Input directory containing log-mel .npy files')
    parser.add_argument('--output', '-o', default='../dataset_features/logmel/metadata/logmel_metadata.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    generate_feature_metadata_csv(feature_dir=args.input, output_csv=args.output)

if __name__ == "__main__":
    main()
