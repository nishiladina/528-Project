#!/usr/bin/env python3
"""
IMU Gesture Visualizer — Flask web app.

Expects data files at:  hw3-data/<gesture>/<gesture>_NN.txt
Each file has a header row (AX,AY,AZ,GX,GY,GZ) followed by CSV lines.
Data is recorded at 100 Hz.

Run:
  pip install flask numpy scipy matplotlib
  python display_data.py
Then open http://127.0.0.1:5000
"""

import os
import io
import base64

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

# GESTURES  = ["up", "down", "left", "right", "test", "captured"]
DATA_DIR  = "hw3-data"
SAMPLE_RATE = 100.0          # Hz — fixed recording rate

# ── Colour palette ─────────────────────
PAL = {
    "ax":    "#2196f3",
    "ay":    "#f44336",
    "az":    "#4caf50",
    "gx":    "#9c27b0",
    "gy":    "#ff9800",
    "gz":    "#009688",
    "bg":    "#ffffff",
    "panel": "#f9f9f9",
}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def load_file(gesture: str, filename: str):
    """
    Load a .txt data file and return (t, ax, ay, az, gx, gy, gz).

    File format:
        AX,AY,AZ,GX,GY,GZ          ← header (case-insensitive, skipped)
        -0.054,0.191,-1.090,...     ← data rows
    Time axis is synthesised from SAMPLE_RATE (100 Hz).
    """
    path = os.path.join(DATA_DIR, gesture, filename)
    data = np.genfromtxt(path, delimiter=",", skip_header=1)

    if data.ndim == 1:          # single-row edge-case
        data = data[np.newaxis, :]

    ax = data[:, 0]
    ay = data[:, 1]
    az = data[:, 2]
    gx = data[:, 3]
    gy = data[:, 4]
    gz = data[:, 5]

    n = len(ax)
    t = np.arange(n) / SAMPLE_RATE   # seconds

    return t, ax, ay, az, gx, gy, gz


def make_timeseries(t, ax, ay, az, gx, gy, gz, title: str) -> tuple[str, str]:
    """Return (accel_b64, gyro_b64) time-domain plots."""

    def _plot(ylabel, series, colors, labels):
        fig, ax_ = plt.subplots(figsize=(9, 2.8), facecolor=PAL["bg"])
        ax_.set_facecolor(PAL["panel"])
        for y, c, lbl in zip(series, colors, labels):
            ax_.plot(t, y, color=c, lw=1.2, label=lbl)
        ax_.set_xlabel("Time (s)", color="#555", fontsize=8)
        ax_.set_ylabel(ylabel, color="#555", fontsize=8)
        ax_.set_title(title, color="#222", fontsize=9, pad=6)
        ax_.tick_params(colors="#555", labelsize=7)
        for spine in ax_.spines.values():
            spine.set_edgecolor("#ddd")
        ax_.grid(True, color="#eee", lw=0.5, ls="--")
        ax_.legend(fontsize=7, facecolor="#fff",
                   edgecolor="#ddd", labelcolor="#333",
                   loc="upper right")
        fig.tight_layout(pad=0.6)
        return fig_to_b64(fig)

    accel_b64 = _plot(
        "Acceleration (g)",
        [ax, ay, az],
        [PAL["ax"], PAL["ay"], PAL["az"]],
        ["AX", "AY", "AZ"],
    )
    gyro_b64 = _plot(
        "Angular rate (°/s)",
        [gx, gy, gz],
        [PAL["gx"], PAL["gy"], PAL["gz"]],
        ["GX", "GY", "GZ"],
    )
    return accel_b64, gyro_b64


def make_spectrograms(t, ax, ay, az, gx, gy, gz, title: str) -> tuple[str, str]:
    """Return (accel_spec_b64, gyro_spec_b64) spectrogram images."""
    fs = SAMPLE_RATE

    def _spec(series_list, labels, ylabel):
        fig, axes = plt.subplots(1, len(series_list),
                                 figsize=(9, 2.8),
                                 facecolor=PAL["bg"])
        if len(series_list) == 1:
            axes = [axes]
        fig.suptitle(title, color="#222", fontsize=9)

        for ax_, y, lbl in zip(axes, series_list, labels):
            ax_.set_facecolor(PAL["panel"])
            nperseg = min(64, max(8, len(y) // 4))
            f, t_spec, Sxx = signal.spectrogram(
                y, fs=fs, nperseg=nperseg,
                noverlap=nperseg // 2
            )
            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            im = ax_.pcolormesh(t_spec, f, Sxx_db,
                                shading="gouraud", cmap="viridis")
            ax_.set_xlabel("Time (s)", color="#555", fontsize=7)
            ax_.set_ylabel("Freq (Hz)", color="#555", fontsize=7)
            ax_.set_title(lbl, fontsize=8)
            ax_.tick_params(colors="#555", labelsize=6)
            for spine in ax_.spines.values():
                spine.set_edgecolor("#ddd")
            cb = fig.colorbar(im, ax=ax_, pad=0.02)
            cb.ax.tick_params(colors="#555", labelsize=6)
            cb.set_label("dB", color="#555", fontsize=6)

        fig.tight_layout(pad=0.6)
        return fig_to_b64(fig)

    accel_spec = _spec([ax, ay, az], ["AX", "AY", "AZ"], "Acceleration (g)")
    gyro_spec  = _spec([gx, gy, gz], ["GX", "GY", "GZ"], "Angular rate (°/s)")
    return accel_spec, gyro_spec


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/files")
def api_files():
    gesture = request.args.get("gesture", "up")
    folder  = os.path.join(DATA_DIR, gesture)
    if not os.path.isdir(folder):
        return jsonify([])
    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".txt")
    )
    return jsonify(files)


@app.route("/api/plot", methods=["POST"])
def api_plot():
    body    = request.json
    gesture = body.get("gesture", "up")
    files   = body.get("files", [])

    results = []
    for fname in files:
        try:
            t, ax, ay, az, gx, gy, gz = load_file(gesture, fname)
        except Exception as e:
            results.append({"file": fname, "error": str(e)})
            continue

        label = fname.replace(".txt", "")
        ts_acc,  ts_gyro = make_timeseries(t, ax, ay, az, gx, gy, gz, label)
        sp_acc,  sp_gyro = make_spectrograms(t, ax, ay, az, gx, gy, gz, label)

        results.append({
            "file":    fname,
            "ts_acc":  ts_acc,
            "ts_gyro": ts_gyro,
            "sp_acc":  sp_acc,
            "sp_gyro": sp_gyro,
        })

    return jsonify(results)


# ── HTML / CSS / JS ───────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IMU Gesture Visualizer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #f5f5f5;
    color: #222;
    font-family: system-ui, sans-serif;
    font-size: 15px;
    min-height: 100vh;
  }

  header {
    padding: 20px 40px;
    background: #fff;
    border-bottom: 1px solid #ddd;
  }

  header h1 { font-size: 1.2rem; font-weight: 600; color: #111; }
  header p  { font-size: 0.82rem; color: #888; margin-top: 3px; }

  .controls {
    display: flex;
    align-items: flex-start;
    gap: 20px;
    padding: 20px 40px;
    background: #fff;
    border-bottom: 1px solid #ddd;
    flex-wrap: wrap;
  }

  .field { display: flex; flex-direction: column; gap: 5px; }

  .field label { font-size: 0.78rem; font-weight: 500; color: #555; }

  select {
    background: #fff;
    color: #222;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 7px 10px;
    font-size: 0.88rem;
    outline: none;
    cursor: pointer;
    min-width: 160px;
  }

  select:focus { border-color: #555; }

  select[multiple] { min-height: 116px; padding: 4px; }

  .hint { font-size: 0.72rem; color: #999; margin-top: 2px; }

  button#go {
    background: #222;
    border: none;
    color: #fff;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 9px 28px;
    margin-top: 20px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
  }

  button#go:hover  { background: #444; }
  button#go:active { background: #111; }
  button#go.loading { background: #aaa; pointer-events: none; }

  #status { font-size: 0.78rem; color: #888; margin-top: 20px; }

  main {
    padding: 32px 40px;
    display: flex;
    flex-direction: column;
    gap: 48px;
  }

  .file-block { animation: fadeUp 0.3s ease both; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .file-block h2 {
    font-size: 1rem;
    font-weight: 600;
    color: #111;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ddd;
  }

  .section-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
    margin-top: 18px;
  }

  .chart-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }

  .chart-wrap {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    overflow: hidden;
  }

  .chart-wrap img { width: 100%; display: block; }

  .error-block {
    background: #fff5f5;
    border: 1px solid #f5c6c6;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #c0392b;
  }

  #empty {
    text-align: center;
    padding: 80px 0;
    font-size: 0.88rem;
    color: #aaa;
  }

  @media (max-width: 700px) {
    header, .controls, main { padding-left: 18px; padding-right: 18px; }
    .chart-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <h1>IMU Gesture Visualizer</h1>
  <p>MPU6050 · 100 Hz · time-domain &amp; spectral analysis</p>
</header>

<div class="controls">
  <div class="field">
    <label>Direction</label>
    <select id="gesture">
      <option value="up">up</option>
      <option value="down">down</option>
      <option value="left">left</option>
      <option value="right">right</option>
      <option value="test">test</option>
      <option value="captured">captured</option>
    </select>
  </div>

  <div class="field">
    <label>File(s)</label>
    <select id="files" multiple></select>
    <div class="hint">hold Ctrl / Cmd to select multiple</div>
  </div>

  <button id="go">GO</button>
  <div id="status"></div>
</div>

<main id="main">
  <div id="empty">select a direction and file(s), then press GO</div>
</main>

<script>
const gestureEl = document.getElementById('gesture');
const filesEl   = document.getElementById('files');
const goBtn     = document.getElementById('go');
const mainEl    = document.getElementById('main');
const statusEl  = document.getElementById('status');

async function loadFiles(gesture) {
  filesEl.innerHTML = '<option disabled>loading…</option>';
  const res  = await fetch(`/api/files?gesture=${gesture}`);
  const list = await res.json();
  filesEl.innerHTML = '';
  if (list.length === 0) {
    filesEl.innerHTML = '<option disabled>no files found</option>';
    return;
  }
  list.forEach(f => {
    const o = document.createElement('option');
    o.value = f; o.textContent = f;
    filesEl.appendChild(o);
  });
  filesEl.options[0].selected = true;
}

gestureEl.addEventListener('change', () => loadFiles(gestureEl.value));
loadFiles(gestureEl.value);

goBtn.addEventListener('click', async () => {
  const gesture  = gestureEl.value;
  const selected = [...filesEl.selectedOptions].map(o => o.value);
  if (selected.length === 0) {
    statusEl.textContent = 'select at least one file';
    return;
  }

  goBtn.classList.add('loading');
  goBtn.textContent = 'LOADING…';
  statusEl.textContent = `fetching ${selected.length} file(s)…`;
  mainEl.innerHTML = '';

  const res     = await fetch('/api/plot', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ gesture, files: selected }),
  });
  const results = await res.json();

  goBtn.classList.remove('loading');
  goBtn.textContent    = 'GO';
  statusEl.textContent = `rendered ${results.length} file(s)`;

  results.forEach((r, idx) => {
    const block = document.createElement('div');
    block.className = 'file-block';
    block.style.animationDelay = `${idx * 0.07}s`;

    if (r.error) {
      block.innerHTML = `
        <h2>${r.file}</h2>
        <div class="error-block">Error: ${r.error}</div>`;
    } else {
      block.innerHTML = `
        <h2>${r.file}</h2>

        <div class="section-label">time domain — acceleration &amp; gyroscope</div>
        <div class="chart-row">
          <div class="chart-wrap"><img src="data:image/png;base64,${r.ts_acc}"  alt="Accel time series"></div>
          <div class="chart-wrap"><img src="data:image/png;base64,${r.ts_gyro}" alt="Gyro time series"></div>
        </div>

        <div class="section-label">spectral analysis — acceleration &amp; gyroscope</div>
        <div class="chart-row">
          <div class="chart-wrap"><img src="data:image/png;base64,${r.sp_acc}"  alt="Accel spectrogram"></div>
          <div class="chart-wrap"><img src="data:image/png;base64,${r.sp_gyro}" alt="Gyro spectrogram"></div>
        </div>`;
    }

    mainEl.appendChild(block);
  });
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)