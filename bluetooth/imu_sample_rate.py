#!/usr/bin/env python3
"""
Estimates the real-world sampling rate of the MPU6050 BLE stream.

Connects to the ESP32, counts incoming samples over a measurement
window, and prints a live running estimate to the terminal.

A "sample" is counted once a complete Accel+Gyro pair has been received.

Usage:
  pip install bleak
  python imu_sample_rate.py
  python imu_sample_rate.py --window 5
"""

import asyncio
import argparse
import re
import time

from bleak import BleakClient, BleakScanner

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE_NAME = "IMU-Stream"
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

ACCEL_RE = re.compile(r"Accel:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)")
GYRO_RE  = re.compile(r"Gyro:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)")

# ── State ─────────────────────────────────────────────────────────────────────
buffer        = ""
pending_accel = None   # holds unpaired Accel line until its Gyro arrives
sample_times  = []     # timestamp of every complete Accel+Gyro pair
window_start  = None
sample_count  = 0


def on_notify(sender, data: bytearray):
    global buffer, pending_accel, sample_times, window_start, sample_count

    buffer += data.decode("utf-8", errors="ignore")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()

        if ACCEL_RE.fullmatch(line):
            pending_accel = line
            continue

        if GYRO_RE.fullmatch(line) and pending_accel is not None:
            pending_accel = None   # pair complete → count as one sample
            now = time.perf_counter()
            if window_start is None:
                window_start = now
            sample_times.append(now)
            sample_count += 1


async def measure(measurement_window: float):
    global window_start, sample_count, sample_times

    print(f"Scanning for '{DEVICE_NAME}'...")
    devices = await BleakScanner.discover(timeout=10.0)
    device  = next((d for d in devices if d.name == DEVICE_NAME), None)

    if device is None:
        print(f"Could not find '{DEVICE_NAME}'. Make sure the ESP32 is powered and advertising.")
        return

    print(f"Found {device.address}. Connecting...")

    async with BleakClient(device) as client:
        print(f"Connected. Measuring in {measurement_window}s windows — press Ctrl+C to stop.\n")
        print(f"  {'Window':<8}  {'Samples':>8}  {'Rate (Hz)':>10}  {'Min gap (ms)':>13}  {'Max gap (ms)':>13}  {'Jitter (ms)':>12}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*13}  {'-'*13}  {'-'*12}")

        await client.start_notify(NUS_TX_UUID, on_notify)

        window_num = 0
        try:
            while True:
                await asyncio.sleep(measurement_window)

                # Snapshot and reset
                times         = sample_times.copy()
                sample_times.clear()
                count         = sample_count
                sample_count  = 0
                window_start  = None
                window_num   += 1

                if len(times) < 2:
                    print(f"  {window_num:<8}  {'--':>8}  {'--':>10}  {'--':>13}  {'--':>13}  {'--':>12}  (not enough samples)")
                    continue

                elapsed = times[-1] - times[0]
                rate    = (count - 1) / elapsed if elapsed > 0 else 0.0

                gaps    = [(times[i+1] - times[i]) * 1000 for i in range(len(times) - 1)]
                min_gap = min(gaps)
                max_gap = max(gaps)
                jitter  = max_gap - min_gap

                print(f"  {window_num:<8}  {count:>8}  {rate:>9.2f}  {min_gap:>12.2f}  {max_gap:>12.2f}  {jitter:>11.2f}")

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            await client.stop_notify(NUS_TX_UUID)


def main():
    parser = argparse.ArgumentParser(description="IMU BLE sampling rate estimator")
    parser.add_argument("--window", type=float, default=3.0,
                        help="Measurement window in seconds (default: 3)")
    args = parser.parse_args()
    asyncio.run(measure(args.window))


if __name__ == "__main__":
    main()