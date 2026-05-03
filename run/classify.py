#!/usr/bin/env python3
"""
IMU Gesture Classifier — MLP Neural Network.

Directory layout expected:
    hw3-data/
        up/       up_01.txt, up_02.txt, ...
        down/     down_01.txt, ...
        left/     left_01.txt, ...
        right/    right_01.txt, ...
        test/     <any_name>.txt          ← unlabelled test files

Each .txt file:
    AX,AY,AZ,GX,GY,GZ          ← header row (skipped)
    -0.054,0.191,-1.090,...     ← data rows at 100 Hz

Usage:
    python classify.py                         # train & save model
    python classify.py --evaluate_training     # score on training data
    python classify.py --evaluate              # predict test/ files
    python classify.py --evaluate --test-dir hw3-data/test
    python classify.py --data-dir hw3-data     # override data root
"""

import argparse
import os
import glob
import pickle

import numpy as np
from scipy import signal, stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Constants ──────────────────────────────────────────────────────────────────
GESTURES             = ["up", "down", "left", "right"]
CLASS_MAP            = {"up": 0, "down": 1, "left": 2, "right": 3}
REV_CLASS_MAP        = {v: k for k, v in CLASS_MAP.items()}

DATA_DIR   = "hw3-data"
TEST_DIR   = os.path.join(DATA_DIR, "test")
MODEL_PATH = "mlp.pkl"
FS         = 100.0           # fixed sample rate (Hz)

ALL_COLS   = list(range(6))  # AX AY AZ GX GY GZ
COL_NAMES  = ["ax", "ay", "az", "gx", "gy", "gz"]


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(filepath: str) -> np.ndarray:
    """
    Load one .txt file and return a 1-D feature vector.

    Key insight from data:
        Up/Down   → GX range >> GZ range  (GX carries the gesture)
        Left/Right → GZ range >> GX range  (GZ carries the gesture)

    Primary features (4):
        gx_range        — how much GX moved (high → up or down)
        gz_range        — how much GZ moved (high → left or right)
        gx_direction    — sign of GX: argmax-before-argmin → down (+), else up (-)
        gz_direction    — sign of GZ: argmax-before-argmin → right (+), else left (-)

    Supporting per-axis features (6 axes × 8 = 48):
        mean, std, min, max, range, RMS, skewness, kurtosis

    Total: 52 features
    """
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    gx = data[:, 3].astype(np.float64)
    gz = data[:, 5].astype(np.float64)
    n  = len(gx)

    # ── Primary features ───────────────────────────────────────────────────────
    gx_range     = float(np.max(gx) - np.min(gx))
    gz_range     = float(np.max(gz) - np.min(gz))
    gx_direction = float(int(np.argmax(gx)) - int(np.argmin(gx))) / n
    gz_direction = float(int(np.argmax(gz)) - int(np.argmin(gz))) / n

    features: list[float] = [gx_range, gz_range, gx_direction, gz_direction]

    # ── Supporting per-axis features ───────────────────────────────────────────
    for col in ALL_COLS:
        x = data[:, col].astype(np.float64)

        mean    = float(np.mean(x))
        std     = float(np.std(x))
        minimum = float(np.min(x))
        maximum = float(np.max(x))
        rng     = maximum - minimum
        rms     = float(np.sqrt(np.mean(x ** 2)))
        skew    = float(stats.skew(x))
        kurt    = float(stats.kurtosis(x))

        features.extend([mean, std, minimum, maximum, rng, rms, skew, kurt])

    return np.array(features, dtype=np.float64)


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(data_dir: str = DATA_DIR):
    """Return (X, y, file_paths) for all labelled gesture folders."""
    X, y, paths = [], [], []
    missing = []

    for gesture in GESTURES:
        folder  = os.path.join(data_dir, gesture)
        pattern = os.path.join(folder, f"{gesture}_*.txt")
        files   = sorted(glob.glob(pattern))

        if not files:
            missing.append(gesture)
            continue

        for fpath in files:
            try:
                feat = extract_features(fpath)
                X.append(feat)
                y.append(CLASS_MAP[gesture])
                paths.append(fpath)
            except Exception as e:
                print(f"  [WARN] Skipping {fpath}: {e}")

    if missing:
        print(f"[WARN] No files found for gesture(s): {missing}")

    return np.array(X), np.array(y), paths


def load_test_files(test_dir: str = TEST_DIR):
    """Return (X, file_paths) for all .txt files in test_dir."""
    files = sorted(glob.glob(os.path.join(test_dir, "*.txt")))
    X, paths = [], []
    for fpath in files:
        try:
            feat = extract_features(fpath)
            X.append(feat)
            paths.append(fpath)
        except Exception as e:
            print(f"  [WARN] Skipping {fpath}: {e}")
    return np.array(X), paths


# ── Model ──────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    StandardScaler + MLP with two hidden layers.

    Architecture:  input(78) → 128 → 64 → 4 classes
    adam optimiser, relu activations, mild L2 regularisation.
    early_stopping uses 15% of training data as a validation split.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,               # L2 regularisation
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
            verbose=False,
        )),
    ])


# ── Train ──────────────────────────────────────────────────────────────────────

def train(data_dir: str = DATA_DIR):
    print(f"\nLoading dataset from '{data_dir}' ...")
    X, y, paths = load_dataset(data_dir)

    if len(X) == 0:
        print("[ERROR] No data loaded. Check that gesture folders exist and contain .txt files.")
        return

    print(f"  {len(X)} samples  |  {X.shape[1]} features  |  {len(set(y))} classes")

    # ── cross-validation estimate (only if enough samples) ────────────────────
    if len(X) >= 10:
        n_splits = min(5, len(X) // len(GESTURES))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(build_pipeline(), X, y,
                                        cv=cv, scoring="accuracy", n_jobs=-1)
            print(f"\n  {n_splits}-fold CV accuracy: "
                  f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── final model on all data ────────────────────────────────────────────────
    print("\nTraining final model on all data ...")
    model = build_pipeline()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    train_acc = model.score(X, y)
    print(f"  Training accuracy : {train_acc:.3f}")
    print(f"  Model saved       → {MODEL_PATH}")


# ── Evaluate on training data ─────────────────────────────────────────────────

def evaluate_on_training(data_dir: str = DATA_DIR):
    """Score the saved model against every labelled training file."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No saved model at '{MODEL_PATH}'. Run training first.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X, y_true, paths = load_dataset(data_dir)
    if len(X) == 0:
        print(f"[ERROR] No training files found in '{data_dir}'.")
        return

    y_pred = model.predict(X)

    # ── per-file table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Evaluation on {len(y_true)} training file(s)")
    print(f"{'='*60}")
    print(f"  {'File':<25} {'True':<8} {'Predicted':<12} Result")
    print(f"  {'-'*52}")

    for fpath, true, pred in zip(paths, y_true, y_pred):
        fname    = os.path.basename(fpath)
        true_lbl = REV_CLASS_MAP[true]
        pred_lbl = REV_CLASS_MAP[pred]
        result   = "PASS" if true == pred else "FAIL"
        print(f"  {fname:<25} {true_lbl:<8} {pred_lbl:<12} {result}")

    accuracy  = (y_true == y_pred).mean()
    n_correct = int((y_true == y_pred).sum())
    print(f"\n  Accuracy : {accuracy:.3f}  ({n_correct}/{len(y_true)} correct)")

    # ── per-class summary ─────────────────────────────────────────────────────
    print(f"\n  {'Class':<8} {'Correct':>8} {'Total':>7} {'Acc':>7}")
    print(f"  {'-'*32}")
    for gesture in GESTURES:
        cls   = CLASS_MAP[gesture]
        mask  = y_true == cls
        if mask.sum() == 0:
            continue
        acc_c = (y_pred[mask] == cls).mean()
        print(f"  {gesture:<8} {int((y_pred[mask]==cls).sum()):>8} "
              f"{int(mask.sum()):>7} {acc_c:>7.3f}")


# ── Evaluate on test data ─────────────────────────────────────────────────────

def evaluate(test_dir: str = TEST_DIR):
    """
    Run the saved model on all .txt files in test_dir.

    If a test file's name starts with a known gesture (e.g. up_test_01.txt),
    the true label is inferred and a PASS/FAIL result is shown.
    Otherwise only the predicted label is printed.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No saved model at '{MODEL_PATH}'. Run training first.")
        return

    if not os.path.isdir(test_dir):
        print(f"[ERROR] Test directory '{test_dir}' not found.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X, paths = load_test_files(test_dir)
    if len(X) == 0:
        print(f"[ERROR] No .txt files found in '{test_dir}'.")
        return

    y_pred = model.predict(X)

    # try to infer ground-truth labels from filenames
    y_true = []
    for fpath in paths:
        fname = os.path.basename(fpath).lower()
        label = None
        for g in GESTURES:
            if fname.startswith(g):
                label = CLASS_MAP[g]
                break
        y_true.append(label)

    has_labels = any(l is not None for l in y_true)

    print(f"\n{'='*60}")
    print(f"  Test results — {len(paths)} file(s) in '{test_dir}'")
    print(f"{'='*60}")

    if has_labels:
        print(f"  {'File':<25} {'True':<8} {'Predicted':<12} Result")
        print(f"  {'-'*52}")
    else:
        print(f"  {'File':<25} {'Predicted':<12}")
        print(f"  {'-'*38}")

    correct, total_labelled = 0, 0
    for fpath, true, pred in zip(paths, y_true, y_pred):
        fname    = os.path.basename(fpath)
        pred_lbl = REV_CLASS_MAP[pred]

        if true is not None:
            true_lbl = REV_CLASS_MAP[true]
            result   = "PASS" if true == pred else "FAIL"
            print(f"  {fname:<25} {true_lbl:<8} {pred_lbl:<12} {result}")
            if true == pred:
                correct += 1
            total_labelled += 1
        else:
            print(f"  {fname:<25} {pred_lbl:<12}")

    if has_labels and total_labelled > 0:
        acc = correct / total_labelled
        print(f"\n  Accuracy : {acc:.3f}  ({correct}/{total_labelled} correct)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLP Neural Network Gesture Classifier")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run predictions on test files in --test-dir (default: hw3-data/test)",
    )
    parser.add_argument(
        "--evaluate_training", action="store_true",
        help="Score the saved model against all labelled training data",
    )
    parser.add_argument(
        "--test-dir", default=TEST_DIR,
        help=f"Directory containing test .txt files (default: {TEST_DIR})",
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR,
        help=f"Root data directory containing gesture sub-folders (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.test_dir)
    elif args.evaluate_training:
        evaluate_on_training(args.data_dir)
    else:
        train(args.data_dir)


if __name__ == "__main__":
    main()