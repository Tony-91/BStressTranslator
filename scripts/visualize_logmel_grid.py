"""
Visualize multiple Log-Mel_Spectrograms slices in a single figure grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse
import math

def find_feature_files(feature_dir: Path):
    """Recursively find all .npy feature files."""
    return list(feature_dir.rglob('*.npy'))

def visualize_logmel_grid(feature_dir: Path, n_samples: int = 6, n_cols: int = 3):
    """Randomly select and plot log-mel slices in a grid."""
    feature_files = find_feature_files(feature_dir)
    
    if not feature_files:
        print(f"No .npy feature files found in {feature_dir}")
        return
    
    sample_files = random.sample(feature_files, min(n_samples, len(feature_files)))
    n_rows = math.ceil(len(sample_files) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axes = axes.flatten()
    
    for ax, f in zip(axes, sample_files):
        arr = np.load(f)
        im = ax.imshow(arr, origin='lower', aspect='auto', cmap='magma')
        ax.set_title(f"{f.parent.name} / {f.name}", fontsize=8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bins")
        fig.colorbar(im, ax=ax, format="%+2.0f dB")
    
    # Hide any unused subplots
    for i in range(len(sample_files), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize multiple Log-Mel Spectrogram slices in a grid")
    parser.add_argument('--input', '-i', required=True, help="Directory containing .npy log-mel features")
    parser.add_argument('--samples', '-n', type=int, default=6, help="Number of slices to visualize")
    parser.add_argument('--cols', '-c', type=int, default=3, help="Number of columns in the grid")
    args = parser.parse_args()
    
    feature_dir = Path(args.input)
    visualize_logmel_grid(feature_dir, n_samples=args.samples, n_cols=args.cols)
