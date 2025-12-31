import json
import random
import csv
from pathlib import Path
from collections import defaultdict

# ---------------- CONFIG ----------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "dataset_features" / "logmel"
OUTPUT_DIR = FEATURE_DIR / "metadata"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ----------------------------------------

def extract_sample_id(filename: str) -> str:
    """
    Extract sample_id from feature filename.
    Example:
      hungry_s05_t01_clean_w10.npy -> hungry_s05_t01
    """
    parts = filename.replace(".npy", "").split("_")
    return "_".join(parts[:3])  # class_sXX_tXX

def main():
    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all feature files
    feature_files = list(FEATURE_DIR.glob("*/*.npy"))
    if not feature_files:
        raise RuntimeError("No .npy feature files found.")

    # Build class map
    classes = sorted({f.parent.name for f in feature_files})
    class_map = {cls: idx for idx, cls in enumerate(classes)}

    # Group files by sample_id
    samples = defaultdict(list)

    for f in feature_files:
        sample_id = extract_sample_id(f.name)
        samples[sample_id].append(f)

    sample_ids = list(samples.keys())
    random.shuffle(sample_ids)

    # Split by sample_id (PREVENTS DATA LEAKAGE)
    n_total = len(sample_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_ids = set(sample_ids[:n_train])
    val_ids = set(sample_ids[n_train:n_train + n_val])
    test_ids = set(sample_ids[n_train + n_val:])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Write dataset_index.csv
    index_path = OUTPUT_DIR / "dataset_index.csv"

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "feature_path",
            "label",
            "label_id",
            "split"
        ])

        for sample_id, files in samples.items():
            split = (
                "train" if sample_id in train_ids else
                "val" if sample_id in val_ids else
                "test"
            )

            for feature_path in files:
                label = feature_path.parent.name
                writer.writerow([
                    sample_id,
                    str(feature_path.relative_to(PROJECT_ROOT)),
                    label,
                    class_map[label],
                    split
                ])

    # Write class_map.json
    with open(OUTPUT_DIR / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    print("✅ dataset_index.csv created")
    print("✅ class_map.json created")
    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples:   {len(val_ids)}")
    print(f"Test samples:  {len(test_ids)}")

if __name__ == "__main__":
    main()
