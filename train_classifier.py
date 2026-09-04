"""
CrunchIQ - Train + evaluate a RandomForest classifier on the acoustic dataset.

With small sample sizes (~10/class) this script is deliberately conservative:
- If there's enough data per class, it uses a stratified train/test split.
- If not, it falls back to Leave-One-Out cross-validation instead of a single
  held-out split, and prints an explicit caveat rather than a misleading
  "accuracy" number.
- The final model is refit on ALL available data and saved for live_classify.py.

Usage:
    python train_classifier.py
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET_FILE = "dataset.npz"
MODEL_FILE = "model.joblib"


def load_dataset():
    try:
        data = np.load(DATASET_FILE, allow_pickle=True)
    except FileNotFoundError:
        print(f"{DATASET_FILE} not found. Run build_dataset.py first.")
        sys.exit(1)
    return data["X"], data["y"]


def main():
    X, y = load_dataset()
    n_samples = len(y)
    class_set = sorted(set(y.tolist()))
    n_classes = len(class_set)
    print(f"Loaded dataset: {n_samples} samples, {n_classes} classes {class_set}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    min_class_count = min(int(np.sum(y == c)) for c in class_set)

    if min_class_count >= 4 and n_samples >= 3 * n_classes:
        # Enough for a stratified held-out test set.
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, stratify=y, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nHeld-out test accuracy: {acc:.2f} (on {len(y_test)} samples)")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("NOTE: with a small dataset, this number is an indicative signal,")
        print("      not a reliable estimate of real-world accuracy. Frame it as")
        print("      proof-of-concept evidence, not a production metric.")
    else:
        # Too small for a held-out split -> Leave-One-Out cross-validation instead.
        print("\nSample size too small for a held-out test split.")
        print("Using Leave-One-Out cross-validation instead (honest, if noisy, estimate).\n")
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        loo = LeaveOneOut()
        scores = cross_val_score(clf, X_scaled, y, cv=loo)
        print(f"Leave-one-out accuracy: {scores.mean():.2f} "
              f"({int(scores.sum())}/{len(scores)} correct)")
        print("NOTE: LOO on ~10 samples/class is a rough signal only - expect this")
        print("      number to move around. If it isn't clearly above chance")
        print("      (1/num_classes), lean on the rule-based fallback for the live demo.")

    # Refit on ALL data for the deployed/demo model.
    clf_final = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf_final.fit(X_scaled, y)

    joblib.dump(
        {"model": clf_final, "scaler": scaler, "classes": class_set},
        MODEL_FILE,
    )
    print(f"\nSaved trained model -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
