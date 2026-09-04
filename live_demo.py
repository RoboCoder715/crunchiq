"""
CrunchVision - Combined live demo.

One break -> record audio + capture a cross-section photo -> run both
models -> print the fused verdict. Timed to comfortably fit under 90
seconds on camera:
  ~2s  countdown
  ~2.5s audio recording
  <1s  webcam photo capture
  <1s  both models run (CPU inference, single short clip + single small image)
  print fused verdict

Usage:
    python live_demo.py
    python live_demo.py --cam_index 1
"""
import argparse
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

try:
    import cv2
except ImportError:
    cv2 = None

import torch
from features import extract_features, extract_key_features
from rule_based import classify_rule_based
from vision_infer import load_vision_model, predict_image
from fusion import fuse_predictions

SAMPLE_RATE = 22050
CLIP_SECONDS = 2.5
AUDIO_MODEL_FILE = "model.joblib"
VISION_MODEL_FILE = "vision_model.pt"
TEMP_WAV = "_live_snap.wav"
TEMP_JPG = "_live_snap.jpg"
CAM_INDEX = 0


def load_acoustic_model():
    try:
        import joblib
        return joblib.load(AUDIO_MODEL_FILE)
    except FileNotFoundError:
        return None


def record_audio(duration=CLIP_SECONDS, sr=SAMPLE_RATE):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def capture_photo(cam_index=CAM_INDEX, out_path=TEMP_JPG, warmup_frames=5):
    if cv2 is None:
        print("  opencv-python not installed - install with:")
        print("  pip install opencv-python --break-system-packages")
        return None
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"  Could not open camera index {cam_index}.")
        return None
    frame = None
    for _ in range(warmup_frames):  # let auto-exposure/focus settle
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()
    if frame is None:
        print("  Failed to capture a frame.")
        return None
    cv2.imwrite(out_path, frame)
    return out_path


def predict_acoustic(wav_path, acoustic_bundle):
    """Rule-based prediction always runs (guaranteed backup); ML model runs too if trained."""
    c, z = extract_key_features(wav_path)
    rb_label, rb_conf = classify_rule_based(c, z)

    if acoustic_bundle is None:
        return rb_label, rb_conf, "rule_based"

    feats = extract_features(wav_path).reshape(1, -1)
    feats_scaled = acoustic_bundle["scaler"].transform(feats)
    proba = acoustic_bundle["model"].predict_proba(feats_scaled)[0]
    pred = acoustic_bundle["model"].classes_[int(np.argmax(proba))]
    conf = float(proba.max())
    return pred, conf, "ml_model"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_index", type=int, default=CAM_INDEX)
    args = parser.parse_args()

    device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    print(f"Compute device : {device_str}")

    acoustic_bundle = load_acoustic_model()
    print(f"Acoustic model : {'trained RandomForest' if acoustic_bundle else 'rule-based fallback only'}")

    if not os.path.exists(VISION_MODEL_FILE):
        print("Vision model  : NOT FOUND (vision_model.pt) - run train_vision.py first.")
        sys.exit(1)
    vision_model, vision_classes, vision_device = load_vision_model(VISION_MODEL_FILE)
    print(f"Vision model   : loaded | classes: {vision_classes} | device: {vision_device}")

    print("\n=== CrunchVision Live Demo ===")
    print("On 'SNAP NOW': break the biscuit near the mic, then immediately hold")
    print("the exposed cross-section up to the camera for the photo.\n")

    try:
        while True:
            cmd = input("Press Enter to run a capture (or 'q' to quit): ").strip().lower()
            if cmd == "q":
                break

            print("Recording audio - SNAP NOW (recording for 2.5 seconds)...")
            t0 = time.time()
            audio = record_audio()
            sf.write(TEMP_WAV, audio, SAMPLE_RATE)
            print(f"  ✓ audio captured ({time.time()-t0:.1f}s)")

            input("  Now hold the broken cross-section up to the camera, then press Enter to photograph...")
            photo_path = capture_photo(cam_index=args.cam_index)
            if photo_path is None:
                print("  Skipping this round - no photo captured.\n")
                continue
            print(f"  ✓ photo captured -> {photo_path}")

            acoustic_label, acoustic_conf, acoustic_source = predict_acoustic(TEMP_WAV, acoustic_bundle)
            vision_label, vision_conf, _ = predict_image(
                photo_path, vision_model, vision_classes, vision_device
            )

            print(f"  [Acoustic:{acoustic_source}] {acoustic_label}  ({acoustic_conf:.0%})")
            print(f"  [Vision]              {vision_label}  ({vision_conf:.0%})")

            result = fuse_predictions(acoustic_label, acoustic_conf, vision_label, vision_conf)
            print(f"  {result.summary()}\n")
    finally:
        for f in (TEMP_WAV, TEMP_JPG):
            if os.path.exists(f):
                os.remove(f)
    print("Demo ended.")


if __name__ == "__main__":
    main()
