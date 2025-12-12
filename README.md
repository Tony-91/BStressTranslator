# BStressTranslator

An audio-based stress detection system that classifies stress levels from voice patterns. Built with Python and machine learning.

## Project Phases

### 1. Dataset Engineering
* Each audio recording was carefully organized into class-specific folders with a consistent naming convention (class_sXX_t01.m4a).
* This approach ensures clear traceability of each recording session and simplifies batch preprocessing with Librosa. By maintaining structured, labeled data from the outset, we optimized the workflow for training on the Jetson Nano, where memory and CPU constraints require efficient, predictable input.
* Attention to reproducibility and edge-AI constraints in real-world audio modeling.
- ATTEN: Offset class imbalance with data augmentation

### 2. Preprocessing Pipeline 
* preprocess_and_slice.py
# Clean
* Mono + 16 kHz: Optimized for Nano 128-core Maxwell GPU and 512 MB GPU memory for spectrogram generation.
* Normalization: Ensures spectrogram amplitude is consistent across classes — critical for model convergence.
* Silence trimming: preserves background noise while removing excessive dead space.
* Output: Each .m4a file becomes a cleaned .wav file.
- Optional configuration adjustments:
TARGET_SR = 16000          # sample rate
TRIM_SILENCE = True        # whether to trim silence
SILENCE_TOP_DB = 20        # threshold for trimming
WINDOW_LENGTH_SEC = 2.0    # slice length
OVERLAP_SEC = 1.0          # slice overlap
On Jetson Nano, reduce overlap or skip silence trimming for speed.
# Slice
Standardized all audio inputs to fixed 2-second windows to simplify deployment on Ai-edge hardware.
2-second overlapping windows capture complete cry patterns, multiply real training data and reduce label noise.
### 3. 
