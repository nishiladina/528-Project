import numpy as np
import re
import glob

SAMPLE_RATE = 33  # Hz


# Load data from text files
def load_data(filename):
    accel = []
    gyro = []

    with open(filename, "r") as f:
        for line in f:
            nums = list(map(float, re.findall(r"-?\d+\.?\d*", line)))

            if line.startswith("Accel"):
                accel.append(nums)
            elif line.startswith("Gyro"):
                gyro.append(nums)

    return np.array(accel), np.array(gyro)


# FFT Features for SVM
def compute_fft_features(signal, fs):
    signal = signal - np.mean(signal)
    N = len(signal)

    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    mag = np.abs(fft_vals) / N

    # Dominant frequency
    idx = np.argmax(mag[1:]) + 1  # skip DC
    dom_freq = freqs[idx]
    dom_mag = mag[idx]

    # Spectral energy
    energy = np.sum(mag**2)

    return [dom_freq, dom_mag, energy]


# Time features for SVM
def compute_time_features(signal):
    idx_max = np.argmax(signal)
    idx_min = np.argmin(signal)
    N = len(signal)
    
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.max(signal) - np.min(signal),  # range
        np.sum(signal**2) / len(signal),   # energy
        float(idx_min < idx_max),         # 1 if min happens first, else 0 for directionality
        (idx_max - idx_min) / N          # normalized order/distance
    ]


# Feature extraction combining time and frequency features from each axis of accel and gyro
def extract_features(accel, gyro):
    features = []

    # Time domain
    for sensor in [accel, gyro]:
        for axis in range(3):
            signal = sensor[:, axis]
            features.extend(compute_time_features(signal))

    # Frequency domain
    for sensor in [accel, gyro]:
        for axis in range(3):
            signal = sensor[:, axis]
            features.extend(compute_fft_features(signal, SAMPLE_RATE))

    return np.array(features)

datasets = ["mark-data", "nishil-data", "david-data"]

# build dataset from all files in the given directory structure
def build_dataset(base_path):
    X = []
    y = []

    gesture_map = {
        "left": 0,
        "right": 1,
        "up": 2,
        "down": 3,
        "right_lean": 4,
        "left_lean": 5,
        "clockwise": 6,
        "counter_clockwise": 7,
        "no_movement": 8
    }
    
    for gesture, label in gesture_map.items():
        for dataset in datasets:
            files = glob.glob(f"{base_path}/{dataset}/{gesture}/*.txt")

            for f in files:
                accel, gyro = load_data(f)
                feat = extract_features(accel, gyro)

                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)


# MAIN function to build dataset and save for SVM training
if __name__ == "__main__":
    X, y = build_dataset("../528-Project/data_collection/data")

    print("Feature shape:", X.shape)  # should be (num_files, ~54)
    print("Labels shape:", y.shape)

    # Save for SVM training
    np.save("X.npy", X)
    np.save("y.npy", y)

    print("Dataset saved!")
