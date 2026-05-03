#!/usr/bin/env python3
"""
Real-time IMU plotter for MPU6050 streamed over BLE (Nordic UART Service)
from an ESP32-S3.

Expected line format (sent over BLE):
  Accel: -0.07, -0.05, -0.96
  Gyro: -1.40, 6.00, 5.42

Accel and Gyro arrive on separate, sequential lines. No temperature channel.

Usage:
  pip install bleak matplotlib
  python plot_imu_ble.py
"""

import asyncio
import re
import threading
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from bleak import BleakClient, BleakScanner

# ── BLE config ────────────────────────────────────────────────────────────────
DEVICE_NAME = "IMU-Stream"
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

# ── Plot config ───────────────────────────────────────────────────────────────
WINDOW_SEC = 5
SAMPLE_HZ  = 100

# ── Line parsers ──────────────────────────────────────────────────────────────
ACCEL_RE = re.compile(r"Accel:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)")
GYRO_RE  = re.compile(r"Gyro:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)")

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":    "#0f1117",
    "panel": "#1a1d27",
    "grid":  "#2a2d3a",
    "ax":    "#4fc3f7",
    "ay":    "#81d4fa",
    "az":    "#b3e5fc",
    "gx":    "#f48fb1",
    "gy":    "#f06292",
    "gz":    "#e91e63",
    "text":  "#e0e0e0",
    "title": "#ffffff",
}

# ── BLE reader ────────────────────────────────────────────────────────────────
class BLEReader:
    """Runs an asyncio BLE loop in a background thread, fills shared deques."""

    def __init__(self, name: str, buf_size: int):
        self.name      = name
        self.lock      = threading.Lock()
        self.status    = "Scanning…"
        self.connected = False
        self._buffer   = ""          # reassembly buffer for chunked BLE packets
        self._pending_accel = None   # unpaired Accel values

        self.t  = deque(maxlen=buf_size)
        self.ax = deque(maxlen=buf_size)
        self.ay = deque(maxlen=buf_size)
        self.az = deque(maxlen=buf_size)
        self.gx = deque(maxlen=buf_size)
        self.gy = deque(maxlen=buf_size)
        self.gz = deque(maxlen=buf_size)

        self._t0     = None
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ble_loop())

    async def _ble_loop(self):
        while True:
            try:
                self.status = f"Scanning for '{self.name}'…"
                devices = await BleakScanner.discover(timeout=10.0)
                device  = next((d for d in devices if d.name == self.name), None)
                if device is None:
                    self.status = f"'{self.name}' not found — retrying…"
                    await asyncio.sleep(3)
                    continue
                self.status = f"Connecting to {device.address}…"
                async with BleakClient(device) as client:
                    self.connected      = True
                    self.status         = f"Connected  {device.address}"
                    self._t0            = time.perf_counter()
                    self._buffer        = ""
                    self._pending_accel = None
                    await client.start_notify(NUS_TX_UUID, self._on_notify)
                    while client.is_connected:
                        await asyncio.sleep(0.5)
            except Exception as e:
                self.connected = False
                self.status    = f"Disconnected — {e}  (retrying…)"
                await asyncio.sleep(3)

    def _on_notify(self, sender, data: bytearray):
        """Called by bleak on each BLE notification (may be a partial line)."""
        self._buffer += data.decode("utf-8", errors="ignore")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()

            m = ACCEL_RE.fullmatch(line)
            if m:
                self._pending_accel = tuple(float(v) for v in m.groups())
                continue

            m = GYRO_RE.fullmatch(line)
            if m and self._pending_accel is not None:
                ax, ay, az = self._pending_accel
                gx, gy, gz = (float(v) for v in m.groups())
                self._pending_accel = None
                now = time.perf_counter() - self._t0
                with self.lock:
                    self.t.append(now)
                    self.ax.append(ax); self.ay.append(ay); self.az.append(az)
                    self.gx.append(gx); self.gy.append(gy); self.gz.append(gz)

    def snapshot(self):
        with self.lock:
            return (
                list(self.t),
                list(self.ax), list(self.ay), list(self.az),
                list(self.gx), list(self.gy), list(self.gz),
            )

# ── Plot helpers ──────────────────────────────────────────────────────────────
def style_axes(ax, ylabel):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text"], labelsize=8)
    ax.yaxis.label.set_color(C["text"])
    ax.xaxis.label.set_color(C["text"])
    ax.set_ylabel(ylabel, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(C["grid"])
    ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="--")

def readout(ax, y):
    return ax.text(1.001, y, "", transform=ax.transAxes,
                   color=C["text"], fontsize=7.5, va="center",
                   fontfamily="monospace")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    buf_size = int(WINDOW_SEC * SAMPLE_HZ * 2)
    reader   = BLEReader(DEVICE_NAME, buf_size)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 7), facecolor=C["bg"])
    fig.canvas.manager.set_window_title("MPU6050 — Real-Time BLE Stream")

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                           left=0.08, right=0.97, top=0.90, bottom=0.08)

    ax_acc  = fig.add_subplot(gs[0])
    ax_gyro = fig.add_subplot(gs[1])

    style_axes(ax_acc,  "Acceleration (g)")
    style_axes(ax_gyro, "Angular rate (°/s)")

    fig.suptitle("MPU6050  Real-Time BLE Stream",
                 color=C["title"], fontsize=13, fontweight="bold")
    status_txt = fig.text(0.5, 0.005, reader.status,
                          ha="center", fontsize=8, color="#888888")

    la_x, = ax_acc.plot([], [],  color=C["ax"],  lw=1.2, label="Accel X")
    la_y, = ax_acc.plot([], [],  color=C["ay"],  lw=1.2, label="Accel Y")
    la_z, = ax_acc.plot([], [],  color=C["az"],  lw=1.2, label="Accel Z")
    lg_x, = ax_gyro.plot([], [], color=C["gx"],  lw=1.2, label="Gyro X")
    lg_y, = ax_gyro.plot([], [], color=C["gy"],  lw=1.2, label="Gyro Y")
    lg_z, = ax_gyro.plot([], [], color=C["gz"],  lw=1.2, label="Gyro Z")

    for a, lines in [(ax_acc,  [la_x, la_y, la_z]),
                     (ax_gyro, [lg_x, lg_y, lg_z])]:
        a.legend(handles=lines, loc="upper left", fontsize=7,
                 facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])

    ax_gyro.set_xlabel("Time (s)", fontsize=9)

    ro_ax = readout(ax_acc,  0.83)
    ro_ay = readout(ax_acc,  0.50)
    ro_az = readout(ax_acc,  0.17)
    ro_gx = readout(ax_gyro, 0.83)
    ro_gy = readout(ax_gyro, 0.50)
    ro_gz = readout(ax_gyro, 0.17)

    def update(_):
        t, ax_, ay_, az_, gx_, gy_, gz_ = reader.snapshot()
        if len(t) < 2:
            status_txt.set_text(reader.status)
            return

        t_now = t[-1]
        t_lo  = t_now - WINDOW_SEC

        def trim(xs):
            return [x for x, ti in zip(xs, t) if ti >= t_lo]

        tv  = [ti for ti in t if ti >= t_lo]
        axv = trim(ax_); ayv = trim(ay_); azv = trim(az_)
        gxv = trim(gx_); gyv = trim(gy_); gzv = trim(gz_)

        la_x.set_data(tv, axv); la_y.set_data(tv, ayv); la_z.set_data(tv, azv)
        lg_x.set_data(tv, gxv); lg_y.set_data(tv, gyv); lg_z.set_data(tv, gzv)

        for a in (ax_acc, ax_gyro):
            a.set_xlim(t_lo, t_now)

        def auto_ylim(ax, *series):
            vals = [v for s in series for v in s]
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max((hi - lo) * 0.15, 0.05)
                ax.set_ylim(lo - pad, hi + pad)

        auto_ylim(ax_acc,  axv, ayv, azv)
        auto_ylim(ax_gyro, gxv, gyv, gzv)

        if ax_:
            ro_ax.set_text(f"AX {ax_[-1]:+7.3f}")
            ro_ay.set_text(f"AY {ay_[-1]:+7.3f}")
            ro_az.set_text(f"AZ {az_[-1]:+7.3f}")
            ro_gx.set_text(f"GX {gx_[-1]:+7.2f}")
            ro_gy.set_text(f"GY {gy_[-1]:+7.2f}")
            ro_gz.set_text(f"GZ {gz_[-1]:+7.2f}")

        status_txt.set_text(reader.status)

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()