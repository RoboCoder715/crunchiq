"""
CrunchIQ - Waveform + spectrogram plots per class, for slides/video.

Picks one representative sample per class and plots waveform + mel
spectrogram side by side, so the acoustic difference is visible before
you even start explaining it. Good for a 5-8 second screen-recorded shot
early in the demo video.

Usage:
    python visualize.py
    python visualize.py --sample data/fresh/fresh_01.wav data/stale/stale_01.wav data/overbaked/overbaked_01.wav
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from features import SAMPLE_RATE

DATA_DIR = "data"
OUT_DIR = "plots"


def pick_one_per_class():
    paths = {}
    if not os.path.isdir(DATA_DIR):
        return paths
    for cls in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        wavs = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(".wav")])
        if wavs:
            paths[cls] = os.path.join(cls_dir, wavs[0])
    return paths


def plot_class_comparison(class_paths, out_path=None):
    n = len(class_paths)
    if n == 0:
        print("No samples found to plot. Record some samples first, or pass --sample paths.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(11, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (cls, path) in enumerate(class_paths.items()):
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)

        librosa.display.waveshow(y, sr=sr, ax=axes[i, 0], color="#3b6ea5")
        axes[i, 0].set_title(f"{cls} - waveform")
        axes[i, 0].set_ylabel("Amplitude")

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[i, 1], cmap="magma")
        axes[i, 1].set_title(f"{cls} - mel spectrogram")
        fig.colorbar(img, ax=axes[i, 1], format="%+2.0f dB")

    plt.tight_layout()
    out_path = out_path or os.path.join(OUT_DIR, "class_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", nargs="+", help="Explicit wav paths (one per class) to plot")
    args = parser.parse_args()

    if args.sample:
        class_paths = {os.path.basename(os.path.dirname(p)): p for p in args.sample}
    else:
        class_paths = pick_one_per_class()

    plot_class_comparison(class_paths)


if __name__ == "__main__":
    main()
