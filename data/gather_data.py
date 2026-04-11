"""
IMU gesture data collector for MPU6050 streamed from an ESP32.

Collects 4 seconds of accelerometer + gyroscope data per gesture,
repeated 20 times, saving each run to a numbered .txt file.

Usage:
  python gather_data.py up
  python gather_data.py down
  python gather_data.py left
  python gather_data.py right
  python gather_data.py up --port /dev/tty.usbserial-11130
  python gather_data.py up --port /dev/tty.usbserial-11130 --baud 115200
"""

import argparse
import re
import sys
import time
import serial
import serial.tools.list_ports
import os

# ── Configuration ─────────────────────────────────────────────────────────────
BAUD_RATE       = 115200
RECORD_DURATION = 4.0    # seconds per gesture capture
NUM_ITERATIONS  = 20     # number of repetitions
# PAUSE_BETWEEN   = 1.0    # seconds of rest between iterations

VALID_GESTURES  = ("up", "down", "left", "right", "test")
DATA_DIR = "data"

# Regex matches both raw ESP_LOGI lines and plain printed lines
LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)"
)

# ── Terminal colours (ANSI) ───────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"

GESTURE_COLOR = {
    "up":    "\033[96m",   # cyan
    "down":  "\033[92m",   # green
    "left":  "\033[93m",   # yellow
    "right": "\033[95m",   # magenta
}

GESTURE_ARROW = {
    "up":    "↑",
    "down":  "↓",
    "left":  "←",
    "right": "→",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_port() -> str:
    """Auto-detect the first USB-serial port."""
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "usbserial" in p.device.lower()]
    if usb:
        return usb[0].device
    if ports:
        return ports[0].device
    print(f"{RED}[ERROR] No serial ports found. Plug in your ESP32 or specify --port.{RESET}",
          file=sys.stderr)
    sys.exit(1)


def parse_line(line: str):
    """Return (ax, ay, az, gx, gy, gz, temp) floats or None."""
    m = LINE_RE.search(line)
    if m:
        return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz", "t"))
    return None

def banner(gesture: str, iteration: int, total: int):
    """Print a clear status banner to the terminal."""
    color  = GESTURE_COLOR.get(gesture, CYAN)
    arrow  = GESTURE_ARROW.get(gesture, "?")
    width  = 52
    border = "─" * width
 
    print(f"\n{BOLD}{color}┌{border}┐{RESET}")
    print(f"{BOLD}{color}│{'':^{width}}│{RESET}")
    print(f"{BOLD}{color}│  {arrow}  Gesture : {gesture.upper():<{width-15}}│{RESET}")
    print(f"{BOLD}{color}│  ◉  Run     : {iteration:>2} / {total:<{width-20}}│{RESET}")
    print(f"{BOLD}{color}│  ⏱  Duration: {RECORD_DURATION:.0f} seconds{'':<{width-24}}│{RESET}")
    print(f"{BOLD}{color}│{'':^{width}}│{RESET}")
    print(f"{BOLD}{color}└{border}┘{RESET}")


def countdown(seconds: int, label: str = "Starting in"):
    """Animated countdown printed on a single line."""
    for i in range(seconds, 0, -1):
        print(f"\r  {YELLOW}{label}: {i}s ...{RESET}   ", end="", flush=True)
        time.sleep(1)
    print(f"\r  {GREEN}▶  GO! — perform the gesture NOW{RESET}         ", flush=True)


def progress_bar(elapsed: float, total: float, width: int = 40) -> str:
    """Return a text progress bar string."""
    ratio  = min(elapsed / total, 1.0)
    filled = int(ratio * width)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(ratio * 100)
    return f"[{bar}] {pct:3d}%  {elapsed:.1f}/{total:.0f}s"


def record_gesture(ser: serial.Serial, gesture: str, iteration: int) -> list[str]:
    """
    Collect RECORD_DURATION seconds of IMU lines from the open serial port (hard coded to 600 samples).
    Returns a list of formatted data strings (one per sample).
    Displays a live progress bar while recording.
    """
    rows = []
    t_start = time.perf_counter()
    # t_end   = t_start + RECORD_DURATION

    print()  # blank line before progress
    samples_recorded = 0
    while samples_recorded < 600:
        now = time.perf_counter()
        elapsed = now - t_start

        # Live progress bar
        bar = progress_bar(elapsed, RECORD_DURATION)
        print(f"\r  {CYAN}{bar}{RESET}", end="", flush=True)

        # if now >= t_end:
        #     break

        try:
            raw  = ser.readline()
            line = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            continue

        parsed = parse_line(line)
        if parsed is None:
            continue

        ax, ay, az, gx, gy, gz, temp = parsed
        ts = now - t_start
        rows.append(f"{ts:.4f},{ax},{ay},{az},{gx},{gy},{gz},{temp}")
        samples_recorded += 1

    # Final bar at 100 %
    bar = progress_bar(RECORD_DURATION, RECORD_DURATION)
    print(f"\r  {GREEN}{bar}{RESET}", flush=True)
    return rows


def save_file(gesture: str, iteration: int, rows: list[str]):
    """Write collected rows to gesture_NN.txt"""
    filename = f"{DATA_DIR}/{gesture}/{gesture}_{iteration:02d}.txt"
    header   = "time,ax,ay,az,gx,gy,gz,temperature"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
    print(f"  {GREEN}✔  Saved {len(rows)} samples → {BOLD}{filename}{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect IMU gesture data (20 × 4 s) and save to numbered .txt files."
    )
    parser.add_argument(
        "gesture",
        choices=VALID_GESTURES,
        help="Gesture label: up | down | left | right",
    )
    parser.add_argument("--port",  default=None,      help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud",  default=BAUD_RATE, type=int, help=f"Baud rate (default {BAUD_RATE})")
    args = parser.parse_args()

    gesture = args.gesture
    port    = args.port or find_port()
    baud    = args.baud

    color = GESTURE_COLOR.get(gesture, CYAN)
    arrow = GESTURE_ARROW.get(gesture, "?")

    print(f"\n{BOLD}{color}  IMU Gesture Recorder  {arrow} {gesture.upper()}{RESET}")
    print(f"  Port : {port}  |  Baud : {baud}")
    print(f"  Runs : {NUM_ITERATIONS}  |  Duration : {RECORD_DURATION:.0f} s each")
    print(f"  Files: {gesture}_01.txt … {gesture}_{NUM_ITERATIONS:02d}.txt")
    print(f"\n{DIM}  Press Ctrl-C at any time to abort.{RESET}\n")

    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            print(f"  {GREEN}Connected to {port} @ {baud} baud{RESET}\n")

            for i in range(0, NUM_ITERATIONS + 1):

                # ── Show banner ────────────────────────────────────────────
                banner(gesture, i, NUM_ITERATIONS)

                # ── Countdown before recording ─────────────────────────────
                countdown(2, label="Get ready")

                # ── Record ────────────────────────────────────────────────
                rows = record_gesture(ser, gesture, i)

                # ── Save ──────────────────────────────────────────────────
                save_file(gesture, i, rows)

                # # ── Pause (skip after last iteration) ─────────────────────
                # if i < NUM_ITERATIONS:
                #     print(f"\n  {DIM}Resting {PAUSE_BETWEEN:.0f}s before next run …{RESET}")
                #     time.sleep(PAUSE_BETWEEN)

    except serial.SerialException as e:
        print(f"\n{RED}[ERROR] Serial connection failed: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}  Aborted by user.{RESET}")
        sys.exit(0)

    print(f"\n{BOLD}{GREEN}  ✔  All {NUM_ITERATIONS} runs complete for '{gesture}'!{RESET}\n")


if __name__ == "__main__":
    main()