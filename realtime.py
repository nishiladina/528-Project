#!/usr/bin/env python3
"""
Real-time IMU Gesture Recognition Pipeline
==========================================
Reads a 6-axis IMU stream from an ESP32 over serial, detects gesture
windows with an energy-based trigger, extracts features, and classifies
with the trained MLP model — all in near real-time.

Usage:
    python realtime.py                          # auto-detect serial port
    python realtime.py --port COM3              # Windows
    python realtime.py --port /dev/ttyUSB0      # Linux / Mac

Then open http://127.0.0.1:5001 in a browser.

Dependencies (beyond the training script):
    pip install flask pyserial numpy scipy scikit-learn
"""

import argparse
import collections
import glob
import json
import os
import pickle
import queue
import threading
import time

import numpy as np
from scipy import signal, stats
from flask import Flask, Response, render_template_string, jsonify, request

# ── Constants ──────────────────────────────────────────────────────────────────
FS               = 100.0          # sample rate (Hz)
MODEL_PATH       = "mlp.pkl"
SAVE_CAPTURES    = True           # save each captured window to captured/capture_NNN.txt

# Gesture segmentation parameters
WINDOW_SAMPLES   = 150           # 1.5 second window at 100 Hz
STEP_SAMPLES     = 10            # slide 100ms between evaluations
PRE_SAMPLES      = 50            # samples before trigger to include
MIN_GESTURE_GAP  = 0.8           # seconds between gesture detections
ENERGY_THRESHOLD = 0.5           # accel variance threshold to trigger detection
QUIET_THRESHOLD  = 0.4           # accel variance threshold to consider "at rest"
QUIET_SAMPLES    = 50            # consecutive quiet samples before re-arming trigger

GESTURES     = ["up", "down", "left", "right"]
CLASS_MAP    = {"up": 0, "down": 1, "left": 2, "right": 3}
REV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
GESTURE_EMOJIS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
GESTURE_COLORS = {"up": "#00d4aa", "down": "#ff6b6b", "left": "#ffd93d", "right": "#6bcbff"}

ALL_COLS  = list(range(6))
ACCEL_COLS = [0, 1, 2]


# ── Feature extraction (mirrors classify.py exactly) ──────────────────────────

def extract_features_from_array(data: np.ndarray) -> np.ndarray:
    """Extract the same 52-feature vector used during training."""

    gx = data[:, 3].astype(np.float64)
    gz = data[:, 5].astype(np.float64)
    n  = len(gx)

    # ── Primary features ───────────────────────────────────────────────────────
    gx_range     = float(np.max(gx) - np.min(gx))
    gz_range     = float(np.max(gz) - np.min(gz))
    gx_direction = float(int(np.argmax(gx)) - int(np.argmin(gx))) / n
    gz_direction = float(int(np.argmax(gz)) - int(np.argmin(gz))) / n

    features: list[float] = [gx_range, gz_range, gx_direction, gz_direction]

    # ── Supporting per-axis features ───────────────────────────────────────────
    for col in ALL_COLS:
        x = data[:, col].astype(np.float64)

        mean    = float(np.mean(x))
        std     = float(np.std(x))
        minimum = float(np.min(x))
        maximum = float(np.max(x))
        rng     = maximum - minimum
        rms     = float(np.sqrt(np.mean(x ** 2)))
        skew    = float(stats.skew(x))
        kurt    = float(stats.kurtosis(x))

        features.extend([mean, std, minimum, maximum, rng, rms, skew, kurt])

    return np.array(features, dtype=np.float64)


# ── Serial reader thread ───────────────────────────────────────────────────────

class SerialReader(threading.Thread):
    """
    Reads lines from the serial port and pushes (timestamp, row) to a queue.
    Also supports a demo/simulation mode when no port is available.
    """

    def __init__(self, port: str | None, baud: int, data_queue: queue.Queue,
                 demo: bool = False):
        super().__init__(daemon=True)
        self.port       = port
        self.baud       = baud
        self.queue      = data_queue
        self.demo       = demo
        self.running    = True
        self.connected  = False
        self.error      = None

    def run(self):
        if self.demo:
            self._run_demo()
        else:
            self._run_serial()

    def _run_serial(self):
        try:
            import serial
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                self.connected = True
                while self.running:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    self._parse_and_push(raw)
        except Exception as e:
            self.error = str(e)
            self.connected = False

    def _run_demo(self):
        """Simulate sensor data: idle baseline with injected gesture bursts."""
        self.connected = True
        rng  = np.random.default_rng(0)
        t    = 0.0
        dt   = 1.0 / FS

        gesture_cycle = collections.deque(GESTURES)

        while self.running:
            # Every ~3 s inject a gesture burst (1 s of data)
            if int(t * FS) % 300 == 0 and t > 0.5:
                gesture = gesture_cycle[0]
                gesture_cycle.rotate(-1)
                burst = self._make_gesture_burst(gesture, rng)
                for row in burst:
                    if not self.running:
                        return
                    self.queue.put((time.time(), row))
                    time.sleep(dt)
            else:
                # idle baseline
                row = np.array([0.04, 0.34, -1.03,
                                -3.5 + rng.normal(0, 0.3),
                                 1.8 + rng.normal(0, 0.3),
                                -4.0 + rng.normal(0, 0.3)])
                row[:3] += rng.normal(0, 0.01, 3)
                self.queue.put((time.time(), row))
                time.sleep(dt)
            t += dt

    @staticmethod
    def _make_gesture_burst(gesture: str, rng) -> list[np.ndarray]:
        """Return ~100 samples that look like a gesture motion."""
        n   = 100
        t_  = np.linspace(0, 1, n)
        env = np.sin(np.pi * t_) * 0.6          # bell-shaped envelope

        profiles = {
            "up":    ( 0.0,  0.5,  0.1,   0,  5,  0),
            "down":  ( 0.0, -0.5, -0.1,   0, -5,  0),
            "left":  (-0.5,  0.0,  0.1,  -8,  0,  0),
            "right": ( 0.5,  0.0, -0.1,   8,  0,  0),
        }
        dx, dy, dz, drx, dry, drz = profiles[gesture]

        rows = []
        for i in range(n):
            e = env[i]
            row = np.array([
                0.04 + dx * e + rng.normal(0, 0.02),
                0.34 + dy * e + rng.normal(0, 0.02),
               -1.03 + dz * e + rng.normal(0, 0.02),
               -3.5  + drx * e + rng.normal(0, 0.5),
                1.8  + dry * e + rng.normal(0, 0.5),
               -4.0  + drz * e + rng.normal(0, 0.5),
            ])
            rows.append(row)
        return rows

    @staticmethod
    def _parse_and_push(line: str):
        pass   # overridden in subclass approach; handled inline in _run_serial


# ── We need a version that actually pushes. Patch after: ─────────────────────

class SerialReaderFixed(SerialReader):
    def _run_serial(self):
        try:
            import serial
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                self.connected = True
                while self.running:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    parts = raw.split(",")
                    if len(parts) == 6:
                        try:
                            row = np.array([float(p) for p in parts])
                            self.queue.put((time.time(), row))
                        except ValueError:
                            pass
        except Exception as e:
            self.error = str(e)
            self.connected = False


# ── Gesture segmenter / classifier ────────────────────────────────────────────

class GestureEngine:
    """
    Maintains a circular buffer of IMU samples.
    Uses an energy trigger on the accelerometer to detect gesture onset,
    then classifies a fixed-length window with the trained model.
    """

    def __init__(self, model, result_queue: queue.Queue):
        self.model        = model
        self.result_queue = result_queue

        self.buffer   = collections.deque(maxlen=WINDOW_SAMPLES + PRE_SAMPLES)
        self.last_ts  = time.time()

        # State machine
        self.armed           = True   # ready to trigger
        self.triggered       = False  # accumulating post-trigger samples
        self.trigger_buf     = []     # samples since trigger
        self.quiet_count     = 0
        self.last_gesture_t  = -999.0

        # Rolling stats for live feed
        self.raw_history     = collections.deque(maxlen=200)  # last 2 s for display

    def push(self, ts: float, row: np.ndarray):
        self.buffer.append(row)
        self.raw_history.append(row.tolist())
        self.last_ts = ts

        accel = row[:3]
        energy = float(np.var(accel))   # variance of current accel vs baseline

        now = time.time()
        gap_ok = (now - self.last_gesture_t) > MIN_GESTURE_GAP

        if self.triggered:
            self.trigger_buf.append(row)

            if len(self.trigger_buf) >= WINDOW_SAMPLES:
                self._classify()
                self.triggered  = False
                self.armed      = False  # wait for quiet before re-arming
                self.quiet_count = 0
                self.trigger_buf = []

        else:
            if not self.armed:
                # wait until sensor returns to rest
                if energy < QUIET_THRESHOLD:
                    self.quiet_count += 1
                    if self.quiet_count >= QUIET_SAMPLES:
                        self.armed = True
                        self.quiet_count = 0
                else:
                    self.quiet_count = 0

            if self.armed and gap_ok and energy > ENERGY_THRESHOLD:
                # fire trigger: prepend buffered pre-samples
                pre = list(self.buffer)[-PRE_SAMPLES:]
                self.trigger_buf = pre[:]
                self.triggered   = True

    def _classify(self):
        data = np.array(self.trigger_buf[:WINDOW_SAMPLES])
        if len(data) < WINDOW_SAMPLES:
            data = np.vstack([
                data,
                np.tile(data[-1], (WINDOW_SAMPLES - len(data), 1))
            ])

        # ── Save capture for debugging ─────────────────────────────────────
        if SAVE_CAPTURES:
            os.makedirs("captured", exist_ok=True)
            existing = len(os.listdir("captured"))
            cap_path = f"captured/capture_{existing:03d}.txt"
            header   = "AX,AY,AZ,GX,GY,GZ"
            np.savetxt(cap_path, data, delimiter=",", header=header, comments="")
            print(f"  [capture] saved {cap_path}  "
                  f"GX range={data[:,3].max()-data[:,3].min():.3f}  "
                  f"GZ range={data[:,5].max()-data[:,5].min():.3f}")

        try:
            feat    = extract_features_from_array(data)
            pred    = int(self.model.predict([feat])[0])
            proba   = self.model.predict_proba([feat])[0].tolist()
            gesture = REV_CLASS_MAP[pred]
            confidence = float(max(proba))
            self.last_gesture_t = time.time()
            self.result_queue.put({
                "gesture":    gesture,
                "confidence": round(confidence, 3),
                "proba":      {REV_CLASS_MAP[i]: round(p, 3) for i, p in enumerate(proba)},
                "ts":         time.time(),
            })
        except Exception as e:
            self.result_queue.put({"error": str(e), "ts": time.time()})


# ── Global state ───────────────────────────────────────────────────────────────

app           = Flask(__name__)
data_queue    = queue.Queue(maxsize=2000)
result_queue  = queue.Queue(maxsize=100)
serial_reader = None
gesture_engine = None
recent_results = collections.deque(maxlen=50)

# SSE subscriber queues
sse_subscribers: list[queue.Queue] = []
sse_lock = threading.Lock()

def broadcast(payload: dict):
    """Push a JSON event to all SSE subscribers."""
    msg = f"data: {json.dumps(payload)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_subscribers.remove(q)


def processing_thread():
    """Drains data_queue → GestureEngine → result_queue → SSE broadcast."""
    while True:
        try:
            ts, row = data_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        gesture_engine.push(ts, row)

        # broadcast raw sample for live chart (throttled: every 2nd sample = 50 Hz)
        if len(gesture_engine.raw_history) % 2 == 0:
            broadcast({"type": "raw", "row": row.tolist(), "ts": ts})

        # broadcast new gesture result if available
        while not result_queue.empty():
            result = result_queue.get_nowait()
            recent_results.appendleft(result)
            broadcast({"type": "gesture", **result})


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/status")
def api_status():
    return jsonify({
        "connected":  serial_reader.connected if serial_reader else False,
        "error":      serial_reader.error if serial_reader else None,
        "demo":       serial_reader.demo if serial_reader else False,
        "model_loaded": gesture_engine is not None,
        "buffer_size": len(gesture_engine.buffer) if gesture_engine else 0,
    })


@app.route("/api/history")
def api_history():
    return jsonify(list(recent_results))


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint."""
    q: queue.Queue = queue.Queue(maxsize=200)
    with sse_lock:
        sse_subscribers.append(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"   # prevent proxy timeouts
        finally:
            with sse_lock:
                if q in sse_subscribers:
                    sse_subscribers.remove(q)

    return Response(generate(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── HTML dashboard ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gesture Recognition</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap');

  :root {
    --bg:      #0a0c10;
    --panel:   #111318;
    --border:  #1e2230;
    --up:      #00d4aa;
    --down:    #ff6b6b;
    --left:    #ffd93d;
    --right:   #6bcbff;
    --dim:     #3a3f52;
    --text:    #c8d0e8;
    --mono:    'Share Tech Mono', monospace;
    --sans:    'Syne', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* scanline overlay */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.07) 2px,
      rgba(0,0,0,0.07) 4px
    );
    pointer-events: none;
    z-index: 9999;
  }

  header {
    display: flex;
    align-items: baseline;
    gap: 18px;
    padding: 22px 32px 18px;
    border-bottom: 1px solid var(--border);
  }

  header h1 {
    font-family: var(--sans);
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #fff;
  }

  .tag {
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--dim);
    display: inline-block;
    margin-right: 6px;
    transition: background 0.3s;
  }
  .status-dot.live { background: var(--up); box-shadow: 0 0 8px var(--up); }
  .status-dot.demo { background: var(--left); box-shadow: 0 0 8px var(--left); }

  #status-label { font-size: 0.72rem; color: var(--dim); }

  .grid {
    display: grid;
    grid-template-columns: 1fr 340px;
    grid-template-rows: auto auto;
    gap: 1px;
    background: var(--border);
    min-height: calc(100vh - 61px);
  }

  .panel {
    background: var(--panel);
    padding: 22px 26px;
  }

  /* ── BIG GESTURE DISPLAY ── */
  .gesture-panel {
    grid-row: 1;
    grid-column: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    min-height: 320px;
    position: relative;
    overflow: hidden;
  }

  .gesture-panel::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,212,170,0.04) 0%, transparent 70%);
    pointer-events: none;
  }

  #gesture-symbol {
    font-family: var(--sans);
    font-size: 7rem;
    font-weight: 800;
    line-height: 1;
    color: #fff;
    transition: transform 0.15s, color 0.2s;
    filter: drop-shadow(0 0 24px currentColor);
  }

  #gesture-name {
    font-family: var(--sans);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text);
  }

  #gesture-conf {
    font-size: 0.72rem;
    color: var(--dim);
    letter-spacing: 0.1em;
  }

  .flash {
    animation: flash-anim 0.4s ease;
  }
  @keyframes flash-anim {
    0%   { transform: scale(1.18); opacity: 0.7; }
    60%  { transform: scale(0.95); opacity: 1; }
    100% { transform: scale(1);    opacity: 1; }
  }

  /* ── PROBABILITY BARS ── */
  .prob-panel {
    grid-row: 2;
    grid-column: 2;
  }

  .prob-label {
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  .prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 11px;
  }

  .prob-gesture {
    width: 48px;
    font-size: 0.72rem;
    color: var(--dim);
    flex-shrink: 0;
  }

  .prob-track {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }

  .prob-fill {
    height: 100%;
    border-radius: 3px;
    width: 0%;
    transition: width 0.25s ease;
  }

  .prob-fill.up    { background: var(--up); }
  .prob-fill.down  { background: var(--down); }
  .prob-fill.left  { background: var(--left); }
  .prob-fill.right { background: var(--right); }

  .prob-pct {
    width: 36px;
    text-align: right;
    font-size: 0.72rem;
    color: var(--dim);
  }

  /* ── LIVE CHART ── */
  .chart-panel {
    grid-row: 1 / 3;
    grid-column: 1;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .chart-label {
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }

  canvas { display: block; width: 100%; }

  /* ── HISTORY LOG ── */
  .log-panel {
    grid-row: 3;
    grid-column: 1 / 3;
    max-height: 200px;
    overflow-y: auto;
    border-top: 1px solid var(--border);
    padding: 14px 26px;
  }

  .log-label {
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }

  .log-entry {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    animation: slideIn 0.25s ease;
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateX(-6px); }
    to   { opacity: 1; transform: translateX(0); }
  }

  .log-time  { color: var(--dim); width: 80px; flex-shrink: 0; }
  .log-dir   { font-family: var(--sans); font-weight: 700; font-size: 0.85rem; width: 80px; }
  .log-conf  { color: var(--dim); width: 60px; }
  .log-bars  { flex: 1; display: flex; gap: 5px; }
  .log-bar   { height: 14px; border-radius: 2px; min-width: 2px; }

  scrollbar-width: thin;
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <h1>GestureNet</h1>
  <span class="tag">IMU · MLP · Real-time</span>
  <span style="margin-left:auto; display:flex; align-items:center;">
    <span class="status-dot" id="dot"></span>
    <span id="status-label">connecting…</span>
  </span>
</header>

<div class="grid">

  <!-- Live signal chart -->
  <div class="panel chart-panel">
    <div class="chart-label">Live Signal — Accelerometer</div>
    <canvas id="ca" height="130"></canvas>
    <div class="chart-label" style="margin-top:10px">Live Signal — Gyroscope</div>
    <canvas id="cg" height="130"></canvas>
    <div id="trigger-bar" style="
      margin-top: 14px;
      font-size: 0.68rem; color: var(--dim); letter-spacing:0.1em;
      display: flex; align-items: center; gap: 10px;">
      <span>ENERGY</span>
      <div style="flex:1; height:4px; background:var(--border); border-radius:2px; overflow:hidden;">
        <div id="energy-fill" style="height:100%; width:0%; background:var(--up); border-radius:2px; transition: width 0.08s;"></div>
      </div>
      <span id="energy-val">0.000</span>
    </div>
  </div>

  <!-- Big gesture display -->
  <div class="panel gesture-panel">
    <div id="gesture-symbol">·</div>
    <div id="gesture-name">WAITING</div>
    <div id="gesture-conf">— no gesture detected —</div>
  </div>

  <!-- Probability bars -->
  <div class="panel prob-panel">
    <div class="prob-label">Class probabilities</div>
    <div class="prob-row">
      <span class="prob-gesture">UP</span>
      <div class="prob-track"><div class="prob-fill up" id="pb-up"></div></div>
      <span class="prob-pct" id="pp-up">0%</span>
    </div>
    <div class="prob-row">
      <span class="prob-gesture">DOWN</span>
      <div class="prob-track"><div class="prob-fill down" id="pb-down"></div></div>
      <span class="prob-pct" id="pp-down">0%</span>
    </div>
    <div class="prob-row">
      <span class="prob-gesture">LEFT</span>
      <div class="prob-track"><div class="prob-fill left" id="pb-left"></div></div>
      <span class="prob-pct" id="pp-left">0%</span>
    </div>
    <div class="prob-row">
      <span class="prob-gesture">RIGHT</span>
      <div class="prob-track"><div class="prob-fill right" id="pb-right"></div></div>
      <span class="prob-pct" id="pp-right">0%</span>
    </div>
  </div>

</div>

<!-- History log -->
<div class="log-panel">
  <div class="log-label">Detection log</div>
  <div id="log"></div>
</div>

<script>
const COLORS = { up:'#00d4aa', down:'#ff6b6b', left:'#ffd93d', right:'#6bcbff' };
const EMOJIS = { up:'↑', down:'↓', left:'←', right:'→' };

// ── Canvas chart setup ────────────────────────────────────────────────────────
const N_POINTS = 200;
const accelBuf = [
  new Float32Array(N_POINTS),
  new Float32Array(N_POINTS),
  new Float32Array(N_POINTS),
];
const gyroBuf = [
  new Float32Array(N_POINTS),
  new Float32Array(N_POINTS),
  new Float32Array(N_POINTS),
];
const ACCEL_COLORS = ['#2196f3','#f44336','#4caf50'];
const GYRO_COLORS  = ['#9c27b0','#ff9800','#009688'];

function drawCanvas(canvasId, bufs, colors, yMin, yMax) {
  const canvas = document.getElementById(canvasId);
  const ctx    = canvas.getContext('2d');
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  canvas.width  = W * devicePixelRatio;
  canvas.height = H * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);

  ctx.fillStyle = '#111318';
  ctx.fillRect(0, 0, W, H);

  // grid lines
  ctx.strokeStyle = '#1e2230';
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  const scaleY = v => H * (1 - (v - yMin) / (yMax - yMin));

  bufs.forEach((buf, bi) => {
    ctx.beginPath();
    ctx.strokeStyle = colors[bi];
    ctx.lineWidth   = 1.2;
    for (let i = 0; i < N_POINTS; i++) {
      const x = (i / (N_POINTS - 1)) * W;
      const y = scaleY(buf[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
}

function pushSample(row) {
  for (let i = 0; i < 3; i++) {
    accelBuf[i].copyWithin(0, 1);
    accelBuf[i][N_POINTS - 1] = row[i];
    gyroBuf[i].copyWithin(0, 1);
    gyroBuf[i][N_POINTS - 1] = row[i + 3];
  }

  // energy meter from accel variance
  let mean = (row[0] + row[1] + row[2]) / 3;
  let vari = ((row[0]-mean)**2 + (row[1]-mean)**2 + (row[2]-mean)**2) / 3;
  document.getElementById('energy-fill').style.width = Math.min(100, vari / 0.5 * 100) + '%';
  document.getElementById('energy-val').textContent = vari.toFixed(3);

  drawCanvas('ca', accelBuf, ACCEL_COLORS, -4, 2);
  drawCanvas('cg', gyroBuf,  GYRO_COLORS,  -750,  750);
}

// ── Gesture result update ─────────────────────────────────────────────────────
function updateGesture(data) {
  const sym  = document.getElementById('gesture-symbol');
  const nm   = document.getElementById('gesture-name');
  const conf = document.getElementById('gesture-conf');
  const col  = COLORS[data.gesture] || '#fff';

  sym.textContent  = EMOJIS[data.gesture] || '?';
  sym.style.color  = col;
  nm.textContent   = data.gesture.toUpperCase();
  conf.textContent = `confidence: ${(data.confidence * 100).toFixed(1)}%`;

  sym.classList.remove('flash');
  void sym.offsetWidth;   // reflow
  sym.classList.add('flash');

  // prob bars
  for (const [g, p] of Object.entries(data.proba || {})) {
    const pct = Math.round(p * 100);
    const el = document.getElementById(`pb-${g}`);
    const pp = document.getElementById(`pp-${g}`);
    if (el) el.style.width = pct + '%';
    if (pp) pp.textContent = pct + '%';
  }

  // log entry
  addLogEntry(data);
}

function addLogEntry(data) {
  const log = document.getElementById('log');
  const d   = new Date(data.ts * 1000);
  const ts  = d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3,'0');
  const col = COLORS[data.gesture] || '#fff';

  const entry = document.createElement('div');
  entry.className = 'log-entry';

  const barsHtml = ['up','down','left','right'].map(g => {
    const pct = Math.round((data.proba?.[g] || 0) * 100);
    return `<div class="log-bar" style="width:${Math.max(pct,2)}px; background:${COLORS[g]};"></div>`;
  }).join('');

  entry.innerHTML = `
    <span class="log-time">${ts}</span>
    <span class="log-dir" style="color:${col}">${EMOJIS[data.gesture]} ${data.gesture.toUpperCase()}</span>
    <span class="log-conf">${(data.confidence*100).toFixed(1)}%</span>
    <div class="log-bars">${barsHtml}</div>`;

  log.insertBefore(entry, log.firstChild);
  while (log.children.length > 40) log.removeChild(log.lastChild);
}

// ── SSE connection ─────────────────────────────────────────────────────────────
let rawThrottle = 0;

function connect() {
  const es = new EventSource('/stream');

  es.onmessage = e => {
    const data = JSON.parse(e.data);
    if (data.type === 'raw') {
      rawThrottle++;
      if (rawThrottle % 2 === 0) pushSample(data.row);
    } else if (data.type === 'gesture') {
      updateGesture(data);
    }
  };

  es.onopen = () => {
    const dot   = document.getElementById('dot');
    const label = document.getElementById('status-label');
    fetch('/api/status').then(r => r.json()).then(s => {
      if (s.demo) {
        dot.className = 'status-dot demo';
        label.textContent = 'DEMO MODE';
      } else {
        dot.className = 'status-dot live';
        label.textContent = 'LIVE · ' + (s.port || '');
      }
    });
  };

  es.onerror = () => {
    document.getElementById('dot').className = 'status-dot';
    document.getElementById('status-label').textContent = 'reconnecting…';
    es.close();
    setTimeout(connect, 2000);
  };
}

// Load history on page load
fetch('/api/history').then(r => r.json()).then(hist => {
  hist.forEach(h => { if (h.gesture) addLogEntry(h); });
});

connect();
</script>
</body>
</html>
"""

# ── Entry point ────────────────────────────────────────────────────────────────

def find_serial_port() -> str | None:
    """Try to auto-detect an ESP32 / CH340 / CP210x serial port."""
    candidates = (
        glob.glob("/dev/ttyUSB*") +
        glob.glob("/dev/ttyACM*") +
        glob.glob("/dev/cu.usbserial*") +
        glob.glob("/dev/cu.SLAB_USBtoUART*") +
        glob.glob("COM[0-9]*")
    )
    return candidates[0] if candidates else None


def main():
    global serial_reader, gesture_engine

    parser = argparse.ArgumentParser(description="Real-time IMU Gesture Recognition")
    parser.add_argument("--port",  default="COM3",  help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud",  default=115200, type=int, help="Baud rate (default: 115200)")
    parser.add_argument("--demo",  action="store_true",
                        help="Run in demo/simulation mode (no serial port needed)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Path to trained model pickle (default: {MODEL_PATH})")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port-flask", default=5001, type=int)
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found at '{args.model}'.")
        print("        Train first:  python classify.py")
        return

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Ensure model supports predict_proba (SVC needs probability=True; MLP always does)
    if not hasattr(model, "predict_proba"):
        print("[ERROR] Model does not support predict_proba. "
              "Re-train with MLPClassifier (classify.py).")
        return

    print(f"✓ Model loaded from '{args.model}'")

    # ── Gesture engine ─────────────────────────────────────────────────────────
    gesture_engine = GestureEngine(model, result_queue)

    # ── Serial reader ──────────────────────────────────────────────────────────
    demo_mode = args.demo
    port      = args.port

    if not demo_mode and port is None:
        port = find_serial_port()
        if port:
            print(f"✓ Auto-detected serial port: {port}")
        else:
            print("[WARN] No serial port found — falling back to DEMO mode.")
            demo_mode = True

    serial_reader = SerialReaderFixed(port, args.baud, data_queue, demo=demo_mode)
    serial_reader.start()

    if demo_mode:
        print("✓ Demo mode active — simulated gestures every ~3 s")
    else:
        print(f"✓ Listening on {port} @ {args.baud} baud")

    # ── Processing thread ──────────────────────────────────────────────────────
    pt = threading.Thread(target=processing_thread, daemon=True)
    pt.start()

    # ── Flask ──────────────────────────────────────────────────────────────────
    print(f"\n  Dashboard → http://{args.host}:{args.port_flask}\n")
    app.run(host=args.host, port=args.port_flask, debug=False, threaded=True)


if __name__ == "__main__":
    main()