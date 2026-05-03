import os
import serial

DIRECTION = "up"
PERSON = "nishil"
PORT = "COM3" # port number
BAUD = 115200       
OUTPUT_DIR = f"data_collection/data/{PERSON}-data/{DIRECTION}" #change folder name to the gesture being recorded
num_files = 20 # number of files to record per gesture
offset = 0 # usually 0, data will start recording at offset + 1
RECORDING_TEST_DATA = False


os.makedirs(OUTPUT_DIR, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=1)

current_file = None

num = offset

try:
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue

        if line.startswith("FILE_START:"):
            print(f"FILE START: {num:02d}")
            
            if RECORDING_TEST_DATA:
                prefix = ""
                if num <= offset + 5:
                    prefix = "up"
                elif num <= offset + 10:
                    prefix = "down"
                elif num <= offset + 15:
                    prefix = "left"
                elif num <= offset + 20:
                    prefix = "right"
                filename = os.path.join(OUTPUT_DIR, f"{prefix}_{DIRECTION}_{num:02d}.txt")
            else:
                filename = os.path.join(OUTPUT_DIR, f"{DIRECTION}_{num:02d}.txt")
           
            current_file = open(filename, "w")
            continue

        if line.startswith("FILE_END:"):
            print(f"FILE END: {num:02d}")
            print("\n\n")
            if current_file is not None:
                current_file.close()
                current_file = None

            if num == num_files + offset:
                break
            continue

        if current_file is not None and line:
            current_file.write(line + "\n")

finally:
    if current_file is not None:
        current_file.close()
    ser.close()