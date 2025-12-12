"""
Script to analyze audio files and generate metadata with improved efficiency and error handling.
Supports multiple audio formats including .m4a and provides detailed file metadata.
"""
import os
import csv
import argparse
import librosa
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path
from typing import Dict, Optional, Any, List

def get_audio_metadata(audio_path: Path, estimate_duration: bool = True, max_duration: float = 10.0) -> Optional[Dict[str, Any]]:
    """Extract metadata from audio file with improved error handling and efficiency.
    
    Args:
        audio_path: Path to the audio file
        estimate_duration: If True, only analyze first max_duration seconds
        max_duration: Maximum duration to analyze if estimate_duration is True
        
    Returns:
        Dictionary containing audio metadata or None if error occurs
    """
    try:
        # Get basic file info
        file_size = os.path.getsize(audio_path)
        
        # Try soundfile first, fall back to pydub if that fails
        try:
            with sf.SoundFile(str(audio_path)) as f:
                duration = f.frames / f.samplerate
                channels = f.channels
                sr = f.samplerate
        except Exception:
            # Fall back to pydub for unsupported formats like .m4a
            try:
                audio = AudioSegment.from_file(str(audio_path))
                duration = len(audio) / 1000.0  # Convert ms to seconds
                channels = audio.channels
                sr = audio.frame_rate
            except Exception as e:
                print(f"Error processing {audio_path} with pydub: {str(e)}")
                return None
        
        # Extract class from parent directory name
        class_name = audio_path.parent.name
        
        # Parse filename components
        stem_parts = audio_path.stem.split('_')
        sample_id = stem_parts[-2] if len(stem_parts) >= 2 else 'unknown'
        task_id = stem_parts[-1] if len(stem_parts) >= 1 else 'unknown'
        
        return {
            'filename': audio_path.name,
            'file_path': str(audio_path.absolute()),
            'sample_rate': sr,
            'duration_seconds': round(duration, 2),
            'channels': channels,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'class': class_name,
            'sample_id': sample_id,
            'task_id': task_id,
            'duration_estimated': estimate_duration and (duration > max_duration)
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def find_audio_files(directory: Path) -> List[Path]:
    """Find all audio files in directory with common extensions."""
    audio_extensions = ['*.m4a', '*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac', '*.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(directory.glob(f'**/{ext}'))
    return audio_files

def generate_metadata_csv(audio_dir: str, output_csv: str, estimate_duration: bool = True) -> None:
    """Generate metadata CSV for all audio files in directory.
    
    Args:
        audio_dir: Directory containing audio files
        output_csv: Path to output CSV file
        estimate_duration: If True, only analyze first 10s of each file
    """
    audio_dir = Path(audio_dir)
    output_path = Path(output_csv)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files and remove duplicates by converting to set and back to list
    audio_files = list(dict.fromkeys(find_audio_files(audio_dir)))
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} unique audio files to process...")
    
    # Get metadata for all files
    metadata = []
    processed_paths = set()  # Track processed file paths
    
    for audio_file in audio_files:
        # Skip if we've already processed this file
        if str(audio_file) in processed_paths:
            continue
            
        meta = get_audio_metadata(audio_file, estimate_duration=estimate_duration)
        if meta:
            # Only add if we haven't seen this file path before
            if meta['file_path'] not in processed_paths:
                metadata.append(meta)
                processed_paths.add(meta['file_path'])
                print(f"Processed: {audio_file}")
    
    if not metadata:
        print("No valid audio files found to process")
        return
    
    # Sort by class and sample ID for consistent output
    metadata.sort(key=lambda x: (x['class'], x.get('sample_id', '')))
    
    # Define output fields
    fieldnames = [
        'filename', 'file_path', 'sample_rate', 'duration_seconds',
        'channels', 'file_size_mb', 'class', 'sample_id', 'task_id',
        'duration_estimated'
    ]
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"Generated metadata for {len(metadata)} files at {output_path}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate metadata for audio files')
    parser.add_argument('--input', '-i', default='../dataset_raw',
                       help='Input directory containing audio files (default: ../dataset_raw)')
    parser.add_argument('--output', '-o', default='../dataset_raw/metadata/audio_metadata.csv',
                       help='Output CSV file path (default: ../dataset_raw/metadata/audio_metadata.csv)')
    parser.add_argument('--full-duration', action='store_true',
                       help='Analyze full duration of audio files (slower)')
    
    args = parser.parse_args()
    
    # Create metadata CSV
    generate_metadata_csv(
        audio_dir=args.input,
        output_csv=args.output,
        estimate_duration=not args.full_duration
    )

if __name__ == "__main__":
    main()
