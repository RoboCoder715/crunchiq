"""
CrunchVision - Paired data collection.

One break of one biscuit produces TWO labeled files: the snap audio and a
photo of the exposed cross-section, saved under matching indices so it's
traceable that both came from the same physical sample:
    data/<class>/<class>_01.wav
    data_vision/<class>/<class>_01.jpg

The two models still TRAIN independently (per the fusion design - no joint
model), but capturing them paired keeps your data collection organized and
means one break session builds both datasets at once.

Usage:
    python capture_pair.py --class fresh --count 10
    python capture_pair.py --class stale --count 10 --cam_index 1
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

try:
    import cv2
except ImportError:
    cv2 = None

SAMPLE_RATE = 22050
CLIP_SECONDS = 3
AUDIO_DIR = "data"
IMAGE_DIR = "data_vision"
CAM_INDEX = 0


def next_index(class_name):
    audio_dir = os.path.join(AUDIO_DIR, class_name)
    os.makedirs(audio_dir, exist_ok=True)
    existing = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    return len(existing) + 1


def record_audio(duration=CLIP_SECONDS, sr=SAMPLE_RATE):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def capture_photo(cam_index=CAM_INDEX, warmup_frames=5):
    if cv2 is None:
        print("  WARNING: opencv-python not installed - skipping photo capture.")
        print("  Install with: pip install opencv-python --break-system-packages")
        return None
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"  WARNING: could not open camera index {cam_index} - skipping photo.")
        return None
    frame = None
    for _ in range(warmup_frames):  # let auto-exposure/focus settle
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_name", required=True,
                         help="Class label, e.g. fresh, stale, overbaked, broken")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--cam_index", type=int, default=CAM_INDEX)
    args = parser.parse_args()

    audio_dir = os.path.join(AUDIO_DIR, args.class_name)
    image_dir = os.path.join(IMAGE_DIR, args.class_name)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    print(f"\n=== Paired capture: {args.class_name} ===")
    print("Protocol: snap the biscuit near the mic, THEN immediately hold the")
    print("broken cross-section up to the camera and press Enter to photograph it.\n")

    start_idx = next_index(args.class_name)
    end_idx = start_idx + args.count - 1

    for i in range(start_idx, end_idx + 1):
        input(f"[{i}/{end_idx}] Press Enter, then snap the biscuit when prompted...")
        print("  Recording in ", end="", flush=True)
        for n in (3, 2, 1):
            print(n, end=" ", flush=True)
            time.sleep(0.6)
        print("-> SNAP NOW!")
        audio = record_audio()
        wav_path = os.path.join(audio_dir, f"{args.class_name}_{i:02d}.wav")
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"  audio saved -> {wav_path}")

        input("  Now hold the cross-section to the camera, press Enter to photograph...")
        frame = capture_photo(cam_index=args.cam_index)
        if frame is not None:
            img_path = os.path.join(image_dir, f"{args.class_name}_{i:02d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"  photo saved -> {img_path}\n")
        else:
            print(f"  photo skipped for sample {i} - add one manually later with a "
                  f"matching filename if you want, or just move on.\n")

    print(f"Done. Paired samples saved under {audio_dir}/ and {image_dir}/")


if __name__ == "__main__":
    main()
