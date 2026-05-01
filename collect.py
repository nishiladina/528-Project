import os
import serial

DIRECTION = "up"
PORT = "COM3" # port number
BAUD = 115200       
OUTPUT_DIR = f"./hw3-data/{DIRECTION}" #change folder name to the gesture being recorded
num_files = 20 # number of files to record per gesture
offset = 0 # usually 0, data will start recording at offset + 1


os.makedirs(OUTPUT_DIR, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=1)

current_file = None

num = offset

try:
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue

        # print(line)

        if line.startswith("FILE_START:"):
            num += 1 # = int(line.split(":")[1])
            print(f"FILE START: {num:02d}")
            
            # uncomment if recording test data
            # prefix = ""
            # if num <= offset + 5:
            #     prefix = "up"
            # elif num <= offset + 10:
            #     prefix = "down"
            # elif num <= offset + 15:
            #     prefix = "left"
            # elif num <= offset + 20:
            #     prefix = "right"
            # filename = os.path.join(OUTPUT_DIR, f"{prefix}_{DIRECTION}_{num:02d}.txt")



            filename = os.path.join(OUTPUT_DIR, f"{DIRECTION}_{num:02d}.txt") # comment out if using test data
            current_file = open(filename, "w")
            continue

        if line.startswith("FILE_END:"):
            print(f"FILE END: {num:02d}")
            print("\n\n")
            if current_file is not None:
                current_file.close()
                current_file = None

            # num = int(line.split(":")[1])
            if num == num_files + offset:
                break
            continue

        if current_file is not None and line:
            # print("-", end="", flush=True)
            current_file.write(line + "\n")

finally:
    if current_file is not None:
        current_file.close()
    ser.close()