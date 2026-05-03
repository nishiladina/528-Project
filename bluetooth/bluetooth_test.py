"""
Shows devices available for connection via Bluetooth. The ESP32 should show up as "IMU-Stream".

Usage:
  pip install bleak
  python bluetooth_test.py
"""
import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for 10 seconds...\n")
    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        print(f"  {d.address}  {d.name or '(no name)'}")

asyncio.run(main())