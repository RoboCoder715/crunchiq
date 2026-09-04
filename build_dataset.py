"""
CrunchIQ - Dataset builder.

Walks data/<class_name>/*.wav, extracts a feature vector for each file,
and produces (X, y, files) arrays, saved to dataset.npz for reuse by
train_classifier.py.

Usage:
    python build_dataset.py
"""
import os
import sys
import numpy as np
from features import extract_features, FEATURE_NAMES

DATA_DIR = "data"
OUT_FILE = "dataset.npz"


def build():
    if not os.path.isdir(DATA_DIR):
        print(f"No '{DATA_DIR}/' folder found. Record samples first with record_samples.py")
        sys.exit(1)

    classes = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    if not classes:
        print(f"No class subfolders found inside '{DATA_DIR}/'.")
        sys.exit(1)

    X, y, files = [], [], []
    print(f"Found classes: {classes}\n")

    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        wavs = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(".wav")])
        print(f"  {cls}: {len(wavs)} files")
        for fname in wavs:
            fpath = os.path.join(cls_dir, fname)
            try:
                feats = extract_features(fpath)
                X.append(feats)
                y.append(cls)
                files.append(fpath)
            except Exception as e:
                print(f"    skipping {fpath}: {e}")

    if not X:
        print("\nNo usable audio files found. Nothing to save.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset built: X shape={X.shape}, classes={sorted(set(y.tolist()))}")

    np.savez(OUT_FILE, X=X, y=y, files=np.array(files), feature_names=np.array(FEATURE_NAMES))
    print(f"Saved -> {OUT_FILE}")

    counts = {c: int((y == c).sum()) for c in classes}
    low = [c for c, n in counts.items() if n < 6]
    if low:
        print(f"\nWarning: classes with <6 samples will make train/test evaluation unreliable: {low}")

    return X, y


if __name__ == "__main__":
    build()
