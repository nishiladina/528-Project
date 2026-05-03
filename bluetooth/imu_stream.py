"""
Prints IMU sensor data streamed over BLE from an ESP32-S3.

Expected line format (sent over BLE):
  Accel: -0.07, -0.05, -0.96
  Gyro: -1.40, 6.00, 5.42

Accel and Gyro lines arrive in pairs and are printed together.

Usage:
  pip install bleak
  python imu_stream.py
"""
import asyncio
import re
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "IMU-Stream"
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device → host (notify)

buffer = ""
pending_accel = None  # holds the most recent unpaired Accel line

ACCEL_RE = re.compile(r"Accel:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)")
GYRO_RE  = re.compile(r"Gyro:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)")


def notification_handler(sender, data: bytearray):
    global buffer, pending_accel
    buffer += data.decode("utf-8", errors="ignore")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()

        m = ACCEL_RE.fullmatch(line)
        if m:
            pending_accel = m.groups()
            continue

        m = GYRO_RE.fullmatch(line)
        if m and pending_accel is not None:
            ax, ay, az = pending_accel
            gx, gy, gz = m.groups()
            print(f"Accel: {ax}, {ay}, {az}  |  Gyro: {gx}, {gy}, {gz}")
            pending_accel = None


async def main():
    print(f"Scanning for '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)

    if device is None:
        print(f"Could not find '{DEVICE_NAME}'. Make sure the ESP32 is powered and advertising.")
        return

    print(f"Found device: {device.address}")

    async with BleakClient(device) as client:
        print("Connected. Subscribing to IMU stream — press Ctrl+C to stop.\n")
        await client.start_notify(NUS_TX_UUID, notification_handler)
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await client.stop_notify(NUS_TX_UUID)
            print("\nDisconnected.")


if __name__ == "__main__":
    asyncio.run(main())