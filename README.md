# BStressTranslator

An audio-based stress detection system that classifies stress levels from voice patterns. Built with Python and machine learning.

## Project Phases

### 1. Dataset Engineering
Collected and structured audio samples for different stress states (hungry, tired). Implemented consistent naming (`{state}_s{XX}_t01.m4a`) and organized data for easy preprocessing. Focused on creating a clean, well-documented dataset as the foundation for model training.

```
dataset_raw/
├── hungry/          # 7 audio samples
└── tired/           # 5 audio samples
```

### 2. Data Preprocessing (Next)
[Will include audio processing, feature extraction, and train/test split details]

### 3. Model Development (Planned)
[Will document model architecture and training approach]

## Quick Start
```bash
git clone [repo-url]
pip install -r requirements.txt
# Run training script
```

## Key Technologies
- Python
- Librosa (audio processing)
- PyTorch/TensorFlow
- Scikit-learn

## Project Structure
```
BStressTranslator/
├── dataset_raw/     # Raw audio samples
├── notebooks/       # Data exploration
├── src/            # Source code
└── README.md
```
