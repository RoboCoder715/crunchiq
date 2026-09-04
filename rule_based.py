"""
CrunchIQ - Rule-based fallback classifier.

Uses just two of the most explainable, strongest-separating features:
  - Spectral centroid (brightness): duller/soggy snaps skew lower,
    crisp/overbaked snaps skew higher.
  - Zero-crossing rate (ZCR): brittle/overbaked snaps are noisiest
    (highest ZCR), soggy breaks are smoothest (lowest ZCR).

This is a guaranteed-to-demo-cleanly backup: no training required. Calibrate
the thresholds once against your own recorded samples before the live demo.

Usage:
    python rule_based.py --calibrate                 # prints per-class stats from data/
    python rule_based.py --classify path/to/clip.wav  # classify a single clip
"""
import argparse
import os
import numpy as np
from features import extract_key_features

DATA_DIR = "data"

# EDIT THESE after running --calibrate on your own recorded samples.
# Defaults below are reasonable starting points for a biscuit snap but
# MUST be checked against your mic/room before the live demo.
CENTROID_LOW = 1800     # below this  -> stale/soggy   (dull, low-frequency thud)
CENTROID_HIGH = 3200    # above this  -> crisp or overbaked (bright, sharp snap)
ZCR_HIGH = 0.12          # high ZCR   -> overbaked/brittle texture (extra "crackle")


def calibrate():
    """Print per-class mean centroid/ZCR from data/ to help you set thresholds above."""
    if not os.path.isdir(DATA_DIR):
        print(f"No '{DATA_DIR}/' folder found. Record samples first.")
        return None

    print(f"{'class':<18}{'centroid_mean(Hz)':>20}{'zcr_mean':>12}")
    stats = {}
    for cls in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        centroids, zcrs = [], []
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            c, z = extract_key_features(os.path.join(cls_dir, fname))
            centroids.append(c)
            zcrs.append(z)
        if centroids:
            stats[cls] = (float(np.mean(centroids)), float(np.mean(zcrs)))
            print(f"{cls:<18}{np.mean(centroids):>20.1f}{np.mean(zcrs):>12.4f}")

    print("\nSet CENTROID_LOW / CENTROID_HIGH / ZCR_HIGH near the midpoints between")
    print("these per-class means, then hardcode them at the top of this file.")
    return stats


def classify_rule_based(centroid, zcr):
    """Simple threshold logic. Returns (label, confidence).
    Labels match the data/<class_name> folder names ('fresh', 'stale',
    'overbaked') so they compare directly against the ML model's output
    and, in CrunchVision, against the vision model's output during fusion.
    """
    if centroid < CENTROID_LOW:
        return "stale", 0.7
    elif centroid >= CENTROID_HIGH and zcr >= ZCR_HIGH:
        return "overbaked", 0.7
    elif centroid >= CENTROID_LOW:
        return "fresh", 0.7
    return "uncertain", 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--classify", type=str, default=None)
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
    elif args.classify:
        c, z = extract_key_features(args.classify)
        label, conf = classify_rule_based(c, z)
        print(f"centroid={c:.1f}Hz  zcr={z:.4f}  -> {label} (confidence {conf:.0%})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
