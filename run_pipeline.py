"""
CrunchVision - Quick-start runner for the full pipeline.

Run this ONE script to go from zero to a working demo on synthetic data.
It chains: synthetic data generation -> acoustic training -> vision training
-> a quick sanity-check of the fusion logic.

For a REAL demo, replace synthetic data with:
  - python capture_pair.py --class fresh --count 15   (and stale, overbaked)
  - then re-run the training steps below.

Usage:
    python run_pipeline.py              # full pipeline on synthetic data
    python run_pipeline.py --skip_gen   # if data already exists
    python run_pipeline.py --epochs 20  # more epochs for vision head
"""
import argparse
import subprocess
import sys
import os


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed (exit {result.returncode}). See output above.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip dataset generation if data already exists")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Vision training epochs (default 15, increase if time allows)")
    parser.add_argument("--backbone", default="mobilenet_v2",
                        choices=["mobilenet_v2", "resnet18"])
    parser.add_argument("--classes", default="fresh stale overbaked",
                        help="Space-separated class names for synthetic generation")
    args = parser.parse_args()

    classes = args.classes.split()

    if not args.skip_gen:
        cls_arg = " ".join(classes)
        run(
            f"{sys.executable} generate_synthetic_dataset.py --count 15 --classes {cls_arg}",
            "Step 1/5: Generate synthetic ACOUSTIC dataset (15 clips/class)"
        )
        run(
            f"{sys.executable} generate_synthetic_vision.py --count 15 --classes {cls_arg}",
            "Step 2/5: Generate synthetic VISION dataset (15 images/class)"
        )
    else:
        print("\n[Skipping generation - using existing data]")

    run(
        f"{sys.executable} build_dataset.py",
        "Step 3/5: Build acoustic feature dataset (dataset.npz)"
    )
    run(
        f"{sys.executable} train_classifier.py",
        "Step 4/5: Train acoustic RandomForest classifier -> model.joblib"
    )
    run(
        f"{sys.executable} train_vision.py --epochs {args.epochs} --backbone {args.backbone}",
        f"Step 5/5: Train vision MobileNetV2 head ({args.epochs} epochs) -> vision_model.pt"
    )

    print(f"\n{'='*60}")
    print("  Pipeline complete. Sanity-checking fusion logic...")
    print(f"{'='*60}")

    # Quick fusion smoke-test (no live hardware needed)
    smoke = (
        "from fusion import fuse_predictions; "
        "r1 = fuse_predictions('fresh', 0.82, 'fresh', 0.75); "
        "r2 = fuse_predictions('fresh', 0.65, 'stale', 0.72); "
        "print('Agreement case:  ', r1.summary()); "
        "print('Disagreement case:', r2.summary())"
    )
    subprocess.run([sys.executable, "-c", smoke])

    print(f"\n{'='*60}")
    print("  ALL DONE — Ready to demo!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Run the live demo:  python live_demo.py")
    print("  2. For a real demo, collect real pairs:")
    print("       python capture_pair.py --class fresh --count 15")
    print("       python capture_pair.py --class stale --count 15")
    print("       python capture_pair.py --class overbaked --count 15")
    print("     Then re-run: python run_pipeline.py --skip_gen --epochs 20")
    print("\n  See VISION_GUIDE.md -> Section 6 for live-vs-prerecorded demo guidance.")


if __name__ == "__main__":
    main()
