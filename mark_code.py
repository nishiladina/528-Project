import numpy as np
import matplotlib.pyplot as plt
import re
import serial
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

PORT = "/dev/cu.SLAB_USBtoUART"
BAUD = 115200
DURATION = 4  # seconds per file
NUM_FILES = 20  # number of files to collect per direction

directions = ["down", "left", "right", "up"]  # Directions used


def parse_data(raw_line):
    pattern = r"([A-G][X-Z]):(-?\d+\.\d+)"
    matches = dict(re.findall(pattern, raw_line))

    if matches:
        ax = float(matches.get("AX", 0))
        ay = float(matches.get("AY", 0))
        az = float(matches.get("AZ", 0))

        gx = float(matches.get("GX", 0))
        gy = float(matches.get("GY", 0))
        gz = float(matches.get("GZ", 0))

        return gx, gy, gz, ax, ay, az
    return None


def create_files():
    # Reads and creates the files needed for each direction
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Serial connected. Starting collection...\n")
    for i in range(0, len(directions)):
        print(f"Begin moving in {directions[i]} direction")
        time.sleep(2)
        for j in range(1, NUM_FILES + 1):
            filename = f"{directions[i]}_{j:02d}.txt"
            print(f"Recording file {j}/{NUM_FILES} of {directions[i]} → {filename}")

            with open(filename, "w") as f:
                start = time.time()
                while time.time() - start < DURATION:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        f.write(line + "\n")
                        f.flush()

            print(f"Done. Waiting 2 seconds before next recording...\n")
            time.sleep(2)
        print(
            f"Finished {directions[i]} direction. Waiting 2 seconds before next recording...\n"
        )
    ser.close()
    print("All files collected!")


def create_graphs():
    for i in [4, 5]:
        for direction in directions:
            imu_vals = np.empty((0, 3))
            accel_vals = np.empty((0, 3))
            filename = direction + "_0" + str(i) + ".txt"
            with open(filename, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if parse_data(line) != None:
                        gx, gy, gz, ax, ay, az = parse_data(line)
                        imu_vals = np.append(imu_vals, [[gx, gy, gz]], axis=0)
                        accel_vals = np.append(accel_vals, [[ax, ay, az]], axis=0)

            timestamps = np.linspace(0, 4, len(imu_vals))

            # filters out gravity
            accel_vals = accel_vals - np.mean(accel_vals, axis=0)

            nfft = len(imu_vals)
            X = np.fft.rfft(imu_vals, n=nfft, axis=0)
            Y = np.fft.rfft(accel_vals, n=nfft, axis=0)

            f = np.fft.rfftfreq(nfft, d=1.0 / 100)

            # This plots time series data for the gyroscope
            plt.figure()
            plt.plot(timestamps, imu_vals[:, 0], label="X")
            plt.plot(timestamps, imu_vals[:, 1], label="Y")
            plt.plot(timestamps, imu_vals[:, 2], label="Z")
            plt.title("Rotation vs. Time " + direction + "_" + str(i))
            plt.xlabel("Time (seconds)")
            plt.ylabel("Rotation (degrees per second)")
            plt.legend()
            plt.show()

            # This plots the time series data for the accelerometer
            plt.figure()
            plt.plot(timestamps, accel_vals[:, 0], label="X")
            plt.plot(timestamps, accel_vals[:, 1], label="Y")
            plt.plot(timestamps, accel_vals[:, 2], label="Z")
            plt.title("Acceleration vs. Time " + direction + "_" + str(i))
            plt.xlabel("Time (seconds)")
            plt.ylabel("Acceleration (gravitational force g (9.8 m/s^2)")
            plt.legend()
            plt.show()

            # this is the spectogram for the gyroscope
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.suptitle("Gyroscope Spectrogram " + direction + "_" + str(i))
            for col, (ax, label) in enumerate(zip(axes, ["X", "Y", "Z"])):
                ax.specgram(
                    imu_vals[:, col], Fs=100, cmap="inferno", NFFT=64, noverlap=32
                )
                ax.set_title(f"Gyro {label}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
            plt.tight_layout()
            plt.show()

            # this is the spectogram for the accelerometer
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.suptitle("Accelerometer Spectrogram " + direction + "_" + str(i))
            for col, (ax, label) in enumerate(zip(axes, ["X", "Y", "Z"])):
                ax.specgram(
                    accel_vals[:, col], Fs=100, cmap="inferno", NFFT=64, noverlap=32
                )
                ax.set_title(f"Accel {label}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
            plt.tight_layout()
            plt.show()

# This prepares data by getting it from the files. It also prepares the labels of the data.
def prep_data():
    X_data = []
    y_labels = []

    for direction in directions:
        for i in range(1, 21):
            filename = f"mark-data/{direction}_{i:02d}.txt"
            imu_vals = np.empty((0, 3))
            accel_vals = np.empty((0, 3))

        
            with open(filename, "r") as f:
                for line in f.readlines():
                    parsed = parse_data(line)
                    if parsed is not None:
                        gx, gy, gz, ax, ay, az = parsed
                        imu_vals = np.append(imu_vals, [[gx, gy, gz]], axis=0)
                        accel_vals = np.append(accel_vals, [[ax, ay, az]], axis=0)

                accel_vals = accel_vals - np.mean(accel_vals, axis=0)

                combined = np.hstack((accel_vals, imu_vals))
                
                #Some files had more samples than others, so I only took the first 475, most had around 570
                combined = combined[224:375]

                X_data.append(combined.flatten())
                print(combined.flatten().shape)
                y_labels.append(direction)
        #same code but for david's data
        
        for i in range(1, 21):
            
            filename = f"david-data/{direction}/{direction}_{i:02d}.txt"
            imu_vals = []
            accel_vals = []

            with open(filename, "r") as f:
                for line in f:
                    nums = list(map(float, re.findall(r"-?\d+\.?\d*", line)))

                    if line.startswith("Accel"):
                        accel_vals.append(nums)
                    elif line.startswith("Gyro"):
                        imu_vals.append(nums)

                accel_vals = accel_vals - np.mean(accel_vals, axis=0)
                

                combined = np.hstack((accel_vals, imu_vals))
                
                #Some files had more samples than others, so I only took the first 475, most had around 570
                #combined = combined[:]

                X_data.append(combined.flatten())
                print(combined.flatten().shape)
                y_labels.append(direction)
    X_data = np.array(X_data)
    y_labels = np.array(y_labels)

    return X_data, y_labels


# this trains the classifer and uses Stratified K-fold with 5 folds to get an average accuracy 
def train_classifier(X_data, y_labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 5)

    fold_accuracies = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_data, y_labels)):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        #Normalizes the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        
        #training and testing the model
        svm = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        
        #Calculate the accuracies for each fold
        acc = np.mean(y_pred == y_test)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold+1}: accuracy = {acc}")

    print(f"Mean accuracy: {np.mean(fold_accuracies)}")
    print(f"Std deviation: {np.std(fold_accuracies)}")
    print(classification_report(all_y_true, all_y_pred, target_names=directions))


    
    
def main():
    #create_files()
    #create_graphs()
    X_data, y_labels = prep_data()
    train_classifier(X_data, y_labels)



if __name__ == "__main__":
    main()
