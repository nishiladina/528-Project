#this is the main file of the code where the genera loop will be running

from collections import deque, Counter
import numpy as np
import time 
import serial

import joblib
from svm_dataset import extract_features

model = joblib.load("svm_model.joblib")

WINDOW_SIZE = 100   
STEP_SIZE   = 15    

PORT = "COM3"       # port number
BAUD = 115200 
ser = serial.Serial(PORT, BAUD, timeout=1)

captionsOn = False
activated = True

label_map = {
    0: "left",
    1: "right",
    2: "up",
    3: "down",
    4: "right_lean",
    5: "left_lean",
    6: "clockwise",
    7: "counter_clockwise",
    8: "no_movement"
}

def read_imu_sensor():
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()

        if not line:
            continue

        try:
            values = list(map(float, line.split(",")))

            if len(values) == 6:
                return values

        except ValueError:
            continue


def simulate_imu_stream(model):
    
    buffer = deque(maxlen=WINDOW_SIZE)
    sample_count = 0
    
    prediction_buffer = deque(maxlen=5)

    while True:
        sample = read_imu_sensor() # your hardware/socket call
        
        if sample is None:
            continue

        buffer.append(sample)
        sample_count += 1

        # predict every STEP_SIZE new samples once buffer is full
        if len(buffer) == WINDOW_SIZE and sample_count % STEP_SIZE == 0:
            window   = np.array(buffer)
            window = np.array(buffer)

            accel = window[:, 0:3]
            gyro = window[:, 3:6]

            features = extract_features(accel, gyro)
            
            # after model predicts:
            probs = model.predict_proba([features])[0]
            pred = model.predict([features])[0]
            confidence = np.max(probs)
            
            if confidence < 0.70:
                label = "no_movement"
            else:
                label = label_map[pred]

            print(label, confidence)

            prediction_buffer.append(label)

            if len(prediction_buffer) == 5:
                majority_label = Counter(prediction_buffer).most_common(1)[0][0]

                if majority_label != "no_movement":
                    print("Gesture:", majority_label)
                    prediction_buffer.clear()
                    buffer.clear()
            




if __name__ == "__main__":
    simulate_imu_stream(model)
