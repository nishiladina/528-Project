import os
import serial

DIRECTION = "no_movement"
PORT = "COM5"       # port number
BAUD = 115200       
OUTPUT_DIR = f"./person-data/{DIRECTION}" # change folder name to the gesture being recorded
num_files = 33     # number of files to record per gesture

os.makedirs(OUTPUT_DIR, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=1)

current_file = None

num = 0

try:
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue

        # print(line)

        if line.startswith("FILE_START:"):
            num += 1 # = int(line.split(":")[1])
            print(f"FILE START: {num:02d}")
            
            # Change up to the gesture being recorded
            filename = os.path.join(OUTPUT_DIR, f"{DIRECTION}_{num:02d}.txt")
            current_file = open(filename, "w")
            continue

        if line.startswith("FILE_END:"):
            print(f"FILE END: {num:02d}")
            print("\n\n")
            if current_file is not None:
                current_file.close()
                current_file = None

            # num = int(line.split(":")[1])
            if num == num_files:
                break
            continue

        if current_file is not None and line:
            # print("-", end="", flush=True)
            current_file.write(line + "\n")

finally:
    if current_file is not None:
        current_file.close()
    ser.close()