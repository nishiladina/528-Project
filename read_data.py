import os
import serial

PORT = "COM3"       # port number
BAUD = 115200       
OUTPUT_DIR = "../up" #change folder name to the gesture being recorded
num_files = 33     # number of files to record per gesture

os.makedirs(OUTPUT_DIR, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=1)

current_file = None

try:
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue

        print(line)

        if line.startswith("FILE_START:"):
            num = int(line.split(":")[1])
            
            # Change up to the gesture being recorded
            filename = os.path.join(OUTPUT_DIR, f"up_{num:02d}.txt")
            current_file = open(filename, "w")
            continue

        if line.startswith("FILE_END:"):
            print("\n\n\n\n\n")
            if current_file is not None:
                current_file.close()
                current_file = None

            num = int(line.split(":")[1])
            if num == num_files:
                break
            continue

        if current_file is not None and line:
            current_file.write(line + "\n")

finally:
    if current_file is not None:
        current_file.close()
    ser.close()
