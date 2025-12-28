from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_CSV = PROJECT_ROOT / "dataset_clean_slices" / "metadata" / "slice_metadata.csv"

# Load metadata
df = pd.read_csv(METADATA_CSV)

# Count slices per class
class_counts = df["class"].value_counts().sort_index()

# Plot
plt.figure()
class_counts.plot(kind="bar")
plt.title("Number of 2-Second Audio Slices per Class")
plt.xlabel("Class")
plt.ylabel("Slice Count")
plt.tight_layout()
plt.show()
