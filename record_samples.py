"""
CrunchIQ - Batch recording utility.

Prompt-and-record loop for building the labeled dataset with minimal friction.
Records fixed-length clips (default 3s) per class and saves them as:
    data/<class_name>/<class_name>_01.wav
    data/<class_name>/<class_name>_02.wav
    ...

Usage:
    python record_samples.py --class fresh --count 10
    python record_samples.py --class stale --count 10
    python record_samples.py --class overbaked --count 10
    python record_samples.py --class broken --count 10   # optional 4th class
"""

import argparse
import os
import sys
import time

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("Install deps first: pip install sounddevice soundfile --break-system-packages")
    sys.exit(1)

SAMPLE_RATE = 22050
CLIP_SECONDS = 3
DATA_DIR = "data"


def next_index(class_dir):
    existing = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
    return len(existing) + 1


def record_one(duration=CLIP_SECONDS, sr=SAMPLE_RATE):
    print("  Recording in ", end="", flush=True)
    for n in (3, 2, 1):
        print(n, end=" ", flush=True)
        time.sleep(0.6)
    print("-> SNAP NOW!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("  done.")
    return audio.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_name", required=True,
                         help="Class label, e.g. fresh, stale, overbaked, broken")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to record")
    parser.add_argument("--duration", type=float, default=CLIP_SECONDS)
    args = parser.parse_args()

    class_dir = os.path.join(DATA_DIR, args.class_name)
    os.makedirs(class_dir, exist_ok=True)

    print(f"\n=== Recording class: {args.class_name} ===")
    print("Protocol: keep mic ~10-15cm away, same snap motion each time, quiet room.\n")

    start_idx = next_index(class_dir)
    end_idx = start_idx + args.count - 1
    for i in range(start_idx, end_idx + 1):
        input(f"[{i}/{end_idx}] Press Enter, then snap the biscuit when prompted...")
        audio = record_one(duration=args.duration)
        fname = f"{args.class_name}_{i:02d}.wav"
        fpath = os.path.join(class_dir, fname)
        sf.write(fpath, audio, SAMPLE_RATE)
        print(f"  Saved -> {fpath}\n")

    print(f"Done. {args.count} samples saved to {class_dir}/")


if __name__ == "__main__":
    main()
