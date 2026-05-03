#!/usr/bin/env python3

import argparse
import collections
import glob
import json
import os
import queue
import threading
import time

import numpy as np
import joblib
from flask import Flask, Response, render_template_string, jsonify
import pyautogui

from extra_tree_dataset import extract_features


# =========================
# Constants
# =========================

FS = 33.0
MODEL_PATH = "run/extra_trees.joblib"

STEP_SAMPLES = 15
WINDOW_SAMPLES = 75

label_map = {
    0: "left",
    1: "right",
    2: "up",
    3: "down",
    4: "right_lean",
    5: "left_lean",
    6: "clockwise",
    7: "counter_clockwise",
    8: "no_movement"
}

GESTURE_EMOJIS = {
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓",
    "right_lean": "↘",
    "left_lean": "↙",
    "clockwise": "⟳",
    "counter_clockwise": "⟲",
    "no_movement": "·"
}


# =========================
# Serial Reader
# =========================

class SerialReader(threading.Thread):
    def __init__(self, port, baud, data_queue):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.queue = data_queue
        self.running = True
        self.connected = False
        self.error = None

    def run(self):
        try:
            import serial

            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                self.connected = True
                print(f"Listening on {self.port} @ {self.baud}")

                while self.running:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()

                    if not line:
                        continue

                    parts = line.split(",")

                    if len(parts) == 6:
                        try:
                            row = np.array([float(p) for p in parts], dtype=np.float64)
                            self.queue.put((time.time(), row))
                        except ValueError:
                            pass

        except Exception as e:
            self.error = str(e)
            self.connected = False
            print("Serial error:", self.error)


# =========================
# Gesture Engine
# =========================

class GestureEngine:
    def __init__(self, model, result_queue):
        self.model = model
        self.result_queue = result_queue

        self.buffer = collections.deque(maxlen=WINDOW_SAMPLES)
        self.prediction_buffer = collections.deque(maxlen=3)
        self.sample_count = 0

        self.raw_history = collections.deque(maxlen=200)
        
        self.activated = False
        self.captionsOn = False
        
        
    def movement(self, label, buffer):
        global captionsOn, activated
        #Look Down - scroll down
        if(self.activated and label == "down"):
            pyautogui.scroll(-10)
        
        #Look Up - scroll up
        elif(self.activated and label == "up"):
            pyautogui.scroll(10)

        #Look left - less volume
        elif(self.activated and label == "left"):
            pyautogui.press('volumedown')


        #Look right - more volume
        elif(self.activated and label == "right"):
            pyautogui.press('volumeup')

        #Lay head on left shoulder - Play/Pause
        elif(self.activated and label == "left_lean"):
            pyautogui.press('space')

        #Lay head on right shoulder - mute for youtube shorts
        elif(self.activated and label == "right_lean"):
            pyautogui.press('m')

        #Activate close captions - Rotate head counterclockwise
        elif(self.activated and label == "counter_clockwise"):
            if(self.captionsOn):
                pyautogui.click('C')
            else:
                pyautogui.click('c')
            self.captionsOn = not self.captionsOn

        #Rotate head clockwise - Activate/Deactivate key
        elif(label == "clockwise"):
            self.activated = not self.activated

        if(label == "None"):
            buffer.popleft()

        else:
            buffer.clear()


    def push(self, ts, row):
        self.buffer.append(row)
        self.raw_history.append(row.tolist())
        self.sample_count += 1

        # Same logic as old live_prediction:
        # predict every STEP_SIZE samples once buffer is full
        if len(self.buffer) == WINDOW_SAMPLES and self.sample_count % STEP_SAMPLES == 0:
            self._predict_window()

    def _predict_window(self):
        data = np.array(self.buffer, dtype=np.float64)

        try:
            accel = data[:, 0:3]
            gyro = data[:, 3:6]

            features = extract_features(accel, gyro)

            pred = int(self.model.predict([features])[0])
            label = label_map[pred]

            self.prediction_buffer.append(label)

            if len(self.prediction_buffer) == 3:
                majority_label = collections.Counter(self.prediction_buffer).most_common(1)[0][0]

                if majority_label != "no_movement":
                    print("Gesture:", majority_label)

                    self.result_queue.put({
                        "gesture": majority_label,
                        "ts": time.time()
                    })
                    
                    self.movement(majority_label, self.prediction_buffer)
                    self.prediction_buffer.clear()
                    self.buffer.clear()
                    self.sample_count = 0

        except Exception as e:
            print("Classification error:", e)
            self.result_queue.put({
                "error": str(e),
                "ts": time.time()
            })


# =========================
# Flask App / Global State
# =========================

app = Flask(__name__)

data_queue = queue.Queue(maxsize=2000)
result_queue = queue.Queue(maxsize=100)

serial_reader = None
gesture_engine = None

recent_results = collections.deque(maxlen=50)
sse_subscribers = []
sse_lock = threading.Lock()


def broadcast(payload):
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
    while True:
        try:
            ts, row = data_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        gesture_engine.push(ts, row)

        broadcast({
            "type": "raw",
            "row": row.tolist(),
            "ts": ts
        })

        while not result_queue.empty():
            result = result_queue.get_nowait()
            recent_results.appendleft(result)
            broadcast({
                "type": "gesture",
                **result
            })


# =========================
# Flask Routes
# =========================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/status")
def api_status():
    return jsonify({
        "connected": serial_reader.connected if serial_reader else False,
        "error": serial_reader.error if serial_reader else None,
        "model_loaded": gesture_engine is not None,
        "buffer_size": len(gesture_engine.buffer) if gesture_engine else 0,
    })


@app.route("/api/history")
def api_history():
    return jsonify(list(recent_results))


@app.route("/stream")
def stream():
    q = queue.Queue(maxsize=200)

    with sse_lock:
        sse_subscribers.append(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with sse_lock:
                if q in sse_subscribers:
                    sse_subscribers.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# =========================
# HTML Dashboard
# =========================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IMU Gesture Recognition</title>

<style>
body {
    margin: 0;
    background: #0a0c10;
    color: #c8d0e8;
    font-family: Arial, sans-serif;
}

header {
    padding: 20px 30px;
    border-bottom: 1px solid #1e2230;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

h1 {
    margin: 0;
    font-size: 22px;
    letter-spacing: 2px;
}

.status-dot {
    width: 10px;
    height: 10px;
    background: #555;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.status-dot.live {
    background: #00d4aa;
    box-shadow: 0 0 10px #00d4aa;
}

.grid {
    display: grid;
    grid-template-columns: 1fr 360px;
    gap: 1px;
    background: #1e2230;
    min-height: calc(100vh - 70px);
}

.panel {
    background: #111318;
    padding: 24px;
}

.chart-panel {
    display: flex;
    flex-direction: column;
    gap: 18px;
}

canvas {
    width: 100%;
    height: 180px;
    background: #0a0c10;
    border: 1px solid #1e2230;
}

.gesture-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

#gesture-symbol {
    font-size: 110px;
    font-weight: bold;
    color: white;
    transition: transform 0.2s;
}

#gesture-name {
    font-size: 26px;
    margin-top: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

#gesture-conf {
    margin-top: 10px;
    color: #777;
}

.flash {
    animation: flash 0.35s ease;
}

@keyframes flash {
    0% { transform: scale(1.25); opacity: 0.7; }
    100% { transform: scale(1); opacity: 1; }
}

.energy-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #777;
    font-size: 13px;
}

.energy-track {
    height: 6px;
    background: #1e2230;
    flex: 1;
    border-radius: 4px;
    overflow: hidden;
}

#energy-fill {
    height: 100%;
    width: 0%;
    background: #00d4aa;
}

.log-panel {
    grid-column: 1 / 3;
    max-height: 220px;
    overflow-y: auto;
    border-top: 1px solid #1e2230;
}

.log-entry {
    padding: 8px 0;
    border-bottom: 1px solid #1e2230;
    display: flex;
    gap: 14px;
}

.log-time {
    color: #777;
    width: 100px;
}

.log-gesture {
    font-weight: bold;
}
</style>
</head>

<body>

<header>
    <h1>GestureNet SVM</h1>
    <div>
        <span class="status-dot" id="dot"></span>
        <span id="status-label">connecting...</span>
    </div>
</header>

<div class="grid">

    <div class="panel chart-panel">
        <div>Accelerometer</div>
        <canvas id="accelCanvas"></canvas>

        <div>Gyroscope</div>
        <canvas id="gyroCanvas"></canvas>

        <div class="energy-wrap">
            <span>ENERGY</span>
            <div class="energy-track">
                <div id="energy-fill"></div>
            </div>
            <span id="energy-val">0.000</span>
        </div>
    </div>

    <div class="panel gesture-panel">
        <div id="gesture-symbol">·</div>
        <div id="gesture-name">WAITING</div>
        <div id="gesture-conf">no gesture detected</div>
    </div>

    <div class="panel log-panel">
        <h3>Detection Log</h3>
        <div id="log"></div>
    </div>

</div>

<script>
const N_POINTS = 200;

const accelBuf = [
    new Float32Array(N_POINTS),
    new Float32Array(N_POINTS),
    new Float32Array(N_POINTS)
];

const gyroBuf = [
    new Float32Array(N_POINTS),
    new Float32Array(N_POINTS),
    new Float32Array(N_POINTS)
];

const EMOJIS = {
    left: "←",
    right: "→",
    up: "↑",
    down: "↓",
    right_lean: "↘",
    left_lean: "↙",
    clockwise: "⟳",
    counter_clockwise: "⟲",
    no_movement: "·"
};

function drawCanvas(canvasId, buffers, yMin, yMax) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;

    canvas.width = W * devicePixelRatio;
    canvas.height = H * devicePixelRatio;

    ctx.scale(devicePixelRatio, devicePixelRatio);
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = "#0a0c10";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#1e2230";
    ctx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * H;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }

    const colors = ["#2196f3", "#f44336", "#4caf50"];

    function scaleY(v) {
        return H * (1 - (v - yMin) / (yMax - yMin));
    }

    buffers.forEach((buf, bi) => {
        ctx.beginPath();
        ctx.strokeStyle = colors[bi];
        ctx.lineWidth = 1.5;

        for (let i = 0; i < N_POINTS; i++) {
            const x = (i / (N_POINTS - 1)) * W;
            const y = scaleY(buf[i]);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
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

    const mean = (row[0] + row[1] + row[2]) / 3;
    const energy = (
        (row[0] - mean) ** 2 +
        (row[1] - mean) ** 2 +
        (row[2] - mean) ** 2
    ) / 3;

    document.getElementById("energy-fill").style.width =
        Math.min(100, energy / 0.5 * 100) + "%";

    document.getElementById("energy-val").textContent = energy.toFixed(3);

    drawCanvas("accelCanvas", accelBuf, -4, 4);
    drawCanvas("gyroCanvas", gyroBuf, -750, 750);
}

function updateGesture(data) {
    if (!data.gesture) return;

    const symbol = document.getElementById("gesture-symbol");
    const name = document.getElementById("gesture-name");
    const conf = document.getElementById("gesture-conf");

    symbol.textContent = EMOJIS[data.gesture] || "?";
    name.textContent = data.gesture;
    conf.textContent = "detected";

    symbol.classList.remove("flash");
    void symbol.offsetWidth;
    symbol.classList.add("flash");

    addLogEntry(data);
}

function addLogEntry(data) {
    const log = document.getElementById("log");
    const d = new Date(data.ts * 1000);
    const ts = d.toTimeString().slice(0, 8);

    const entry = document.createElement("div");
    entry.className = "log-entry";

    entry.innerHTML = `
        <span class="log-time">${ts}</span>
        <span class="log-gesture">${EMOJIS[data.gesture] || "?"} ${data.gesture}</span>
    `;

    log.insertBefore(entry, log.firstChild);

    while (log.children.length > 40) {
        log.removeChild(log.lastChild);
    }
}

function connect() {
    const es = new EventSource("/stream");

    es.onopen = () => {
        const dot = document.getElementById("dot");
        const label = document.getElementById("status-label");

        dot.className = "status-dot live";
        label.textContent = "LIVE";
    };

    es.onmessage = e => {
        const data = JSON.parse(e.data);

        if (data.type === "raw") {
            pushSample(data.row);
        }

        if (data.type === "gesture") {
            updateGesture(data);
        }
    };

    es.onerror = () => {
        document.getElementById("dot").className = "status-dot";
        document.getElementById("status-label").textContent = "reconnecting...";
        es.close();
        setTimeout(connect, 2000);
    };
}

connect();
</script>

</body>
</html>
"""


# =========================
# Main
# =========================

def find_serial_port():
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baud", default=115200, type=int)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-flask", default=5001, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return

    model = joblib.load(args.model)
    print(f"Loaded model from {args.model}")

    gesture_engine = GestureEngine(model, result_queue)

    serial_reader = SerialReader(args.port, args.baud, data_queue)
    serial_reader.start()

    pt = threading.Thread(target=processing_thread, daemon=True)
    pt.start()

    print(f"Dashboard: http://{args.host}:{args.port_flask}")
    app.run(host=args.host, port=args.port_flask, debug=False, threaded=True)


if __name__ == "__main__":
    main()