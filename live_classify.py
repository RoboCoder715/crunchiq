"""
CrunchIQ - Live classify loop, built for the on-camera demo.

Records a fresh snap from the mic, extracts features, and predicts using
BOTH the trained RandomForest (if model.joblib exists) and the rule-based
fallback, side by side. Each cycle is a couple of seconds of recording plus
near-instant feature extraction/prediction, so this comfortably fits inside
a 40-second live demo segment.

Usage:
    python live_classify.py
"""
import os
import sys
import time
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("Install deps first: pip install sounddevice soundfile --break-system-packages")
    sys.exit(1)

from features import extract_features, extract_key_features
from rule_based import classify_rule_based

SAMPLE_RATE = 22050
CLIP_SECONDS = 2.5
MODEL_FILE = "model.joblib"
TEMP_WAV = "_live_clip.wav"


def load_model():
    try:
        import joblib
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return None


def record_clip(duration=CLIP_SECONDS, sr=SAMPLE_RATE):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def main():
    model_bundle = load_model()
    if model_bundle:
        print(f"Loaded trained model. Classes: {model_bundle['classes']}")
    else:
        print("No trained model found (model.joblib) - running rule-based fallback only.")

    print("\n=== CrunchIQ Live Demo ===")
    print("Snap the biscuit near the mic when prompted.\n")

    try:
        while True:
            cmd = input("Press Enter to record a snap (or 'q' to quit): ").strip().lower()
            if cmd == "q":
                break

            print("Recording... ", end="", flush=True)
            t0 = time.time()
            audio = record_clip()
            sf.write(TEMP_WAV, audio, SAMPLE_RATE)
            print(f"done ({time.time()-t0:.1f}s)")

            # Rule-based prediction: always runs, guaranteed backup.
            c, z = extract_key_features(TEMP_WAV)
            rb_label, rb_conf = classify_rule_based(c, z)
            print(f"  [Rule-based]  centroid={c:.0f}Hz  zcr={z:.3f}  -> {rb_label}  ({rb_conf:.0%})")

            # ML model prediction, if trained.
            if model_bundle:
                feats = extract_features(TEMP_WAV).reshape(1, -1)
                feats_scaled = model_bundle["scaler"].transform(feats)
                proba = model_bundle["model"].predict_proba(feats_scaled)[0]
                pred = model_bundle["model"].classes_[int(np.argmax(proba))]
                conf = float(proba.max())
                print(f"  [ML model]    -> {pred}  ({conf:.0%} confidence)")

            print()
    finally:
        if os.path.exists(TEMP_WAV):
            os.remove(TEMP_WAV)
    print("Demo ended.")


if __name__ == "__main__":
    main()
