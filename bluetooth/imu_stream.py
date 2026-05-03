"""
Prints IMU sensor data streamed over BLE from an ESP32-S3.

Expected line format (sent over BLE):
  0.039,0.883,-0.077,57.969,-6.115,1.344

Usage:
  pip install bleak
  python imu_stream.py
"""
import asyncio
import re
from bleak import BleakClient, BleakScanner

DEVICE_NAME   = "IMU-Stream"
NUS_TX_UUID   = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device → host (notify)

buffer = ""


LINE_RE = re.compile(
    r"(?P<ax>[-\d.]+),(?P<ay>[-\d.]+),(?P<az>[-\d.]+),"
    r"(?P<gx>[-\d.]+),(?P<gy>[-\d.]+),(?P<gz>[-\d.]+)"
)

def notification_handler(sender, data: bytearray):
    global buffer
    buffer += data.decode("utf-8", errors="ignore")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if LINE_RE.fullmatch(line):
            print(line)

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