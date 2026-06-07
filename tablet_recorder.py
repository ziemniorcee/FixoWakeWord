"""Serve a USB-friendly tablet UI for collecting wake-word recordings."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SAMPLE_RATE = 16_000
NEGATIVE_CHUNK_SECONDS = 3
HARD_NEGATIVE_CHUNK_SECONDS = 2
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDERS = {
    "positive": BASE_DIR / "data/positive_real",
    "negative": BASE_DIR / "data/negative_real",
    "hard_negative": BASE_DIR / "data/hard_negative_real",
}
INDEX_PATTERN = re.compile(r"_(\d{4})(?:_\d{3})?\.wav$")
SAVE_LOCK = threading.Lock()


HTML = r"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fikso sample recorder</title>
  <style>
    :root { font-family: system-ui, sans-serif; color: #172033; background: #eef2f7; }
    body { margin: 0; padding: 18px; }
    main { max-width: 680px; margin: auto; background: white; border-radius: 24px; padding: 24px; box-shadow: 0 8px 30px #0002; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    p { color: #526070; }
    .modes { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 20px 0; }
    button, input { font: inherit; }
    .mode { border: 2px solid #cad3df; background: white; border-radius: 14px; padding: 14px; font-weight: 700; }
    .mode.active { border-color: #235bd8; color: #235bd8; background: #eef4ff; }
    .mode:disabled { opacity: 0.55; }
    label { display: block; margin: 12px 0; font-weight: 700; }
    input { width: 76px; margin-left: 8px; padding: 8px; border: 1px solid #cad3df; border-radius: 8px; }
    #record { width: 100%; min-height: 120px; border: 0; border-radius: 20px; background: #235bd8; color: white; font-size: 30px; font-weight: 800; margin-top: 16px; }
    #record.recording { background: #d62d3e; }
    #progress { height: 14px; overflow: hidden; border-radius: 999px; background: #dfe6ef; margin: 16px 0 0; }
    #progressBar { width: 0%; height: 100%; border-radius: inherit; background: #235bd8; transition: width linear; }
    #progress.recording #progressBar { background: #d62d3e; }
    #status { min-height: 48px; padding: 12px; border-radius: 12px; background: #f5f7fa; font-weight: 650; }
    #stats { font-size: 18px; }
  </style>
</head>
<body>
<main>
  <h1>Fikso sample recorder</h1>
  <p>Nagrania z mikrofonu tabletu zapisujÄ… siÄ™ od razu na komputerze.</p>
  <div class="modes">
    <button class="mode active" data-kind="positive">Fikso</button>
    <button class="mode" data-kind="negative">TĹ‚o / negatywy</button>
    <button class="mode" data-kind="hard_negative">Trudne negatywy</button>
  </div>
  <label>Czas nagrania <input id="seconds" type="number" min="1" max="300" step="1" value="2"> s</label>
  <p id="hint">Kliknij start raz. Recorder bedzie robil kolejne dwusekundowe probki Fikso, az klikniesz stop.</p>
  <button id="record">Nagraj</button>
  <div id="progress"><div id="progressBar"></div></div>
  <p id="status">Gotowe.</p>
  <p id="stats"></p>
</main>
<script>
let kind = "positive";
let recording = false;
let stopRequested = false;
const recordButton = document.querySelector("#record");
const secondsInput = document.querySelector("#seconds");
const statusBox = document.querySelector("#status");
const statsBox = document.querySelector("#stats");
const hint = document.querySelector("#hint");
const progress = document.querySelector("#progress");
const progressBar = document.querySelector("#progressBar");
const modeButtons = Array.from(document.querySelectorAll(".mode"));

modeButtons.forEach(button => button.onclick = () => {
  if (recording) return;
  kind = button.dataset.kind;
  modeButtons.forEach(b => b.classList.toggle("active", b === button));
  secondsInput.value = kind === "positive" || kind === "hard_negative" ? 2 : 60;
  hint.textContent = kind === "positive"
    ? "Kliknij start raz. Recorder bedzie robil kolejne dwusekundowe probki Fikso, az klikniesz stop."
    : kind === "hard_negative"
      ? "Kliknij start raz. Recorder bedzie robil kolejne dwusekundowe trudne negatywy, az klikniesz stop."
      : "Nagraj zwykla rozmowe lub dzwieki pomieszczenia bez slowa Fikso. Serwer potnie nagranie na fragmenty.";
});

async function refreshStats() {
  const response = await fetch("/api/stats");
  const stats = await response.json();
  statsBox.textContent = `Zapisane: Fikso ${stats.positive}, negatywy ${stats.negative}, trudne ${stats.hard_negative}`;
}

function wavBlob(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => view.setInt16(44 + index * 2, Math.max(-1, Math.min(1, sample)) * 0x7fff, true));
  return new Blob([buffer], {type: "audio/wav"});
}

function resample(input, fromRate) {
  if (fromRate === 16000) return input;
  const output = new Float32Array(Math.round(input.length * 16000 / fromRate));
  for (let i = 0; i < output.length; i++) {
    const position = i * fromRate / 16000;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = position - left;
    output[i] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

async function capture(seconds) {
  const stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = event => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  source.connect(processor);
  processor.connect(context.destination);
  await new Promise(resolve => setTimeout(resolve, seconds * 1000));
  processor.disconnect(); source.disconnect(); stream.getTracks().forEach(track => track.stop()); await context.close();
  const joined = new Float32Array(chunks.reduce((size, chunk) => size + chunk.length, 0));
  let offset = 0;
  chunks.forEach(chunk => { joined.set(chunk, offset); offset += chunk.length; });
  return wavBlob(resample(joined, context.sampleRate), 16000);
}

class CaptureSession {
  constructor() {
    this.stream = null;
    this.context = null;
    this.source = null;
    this.processor = null;
    this.chunks = [];
    this.collecting = false;
  }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
    this.context = new AudioContext();
    this.source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = event => {
      if (this.collecting) this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
  }
  async record(seconds) {
    this.chunks = [];
    this.collecting = true;
    await new Promise(resolve => setTimeout(resolve, seconds * 1000));
    this.collecting = false;
    const joined = new Float32Array(this.chunks.reduce((size, chunk) => size + chunk.length, 0));
    let offset = 0;
    this.chunks.forEach(chunk => { joined.set(chunk, offset); offset += chunk.length; });
    return wavBlob(resample(joined, this.context.sampleRate), 16000);
  }
  async stop() {
    if (this.processor) this.processor.disconnect();
    if (this.source) this.source.disconnect();
    if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    if (this.context) await this.context.close();
  }
}

function isSeriesMode() {
  return kind === "positive" || kind === "hard_negative";
}

function setBusy(value) {
  recording = value;
  if (value) {
    recordButton.classList.add("recording");
    modeButtons.forEach(button => button.disabled = true);
    secondsInput.disabled = isSeriesMode();
  } else {
    recordButton.classList.remove("recording");
    recordButton.textContent = "Nagraj";
    modeButtons.forEach(button => button.disabled = false);
    secondsInput.disabled = false;
  }
}

function animateProgress(seconds) {
  progress.classList.add("recording");
  progressBar.style.transition = "none";
  progressBar.style.width = "0%";
  progressBar.offsetHeight;
  progressBar.style.transition = `width ${seconds}s linear`;
  progressBar.style.width = "100%";
}

function resetProgress() {
  progress.classList.remove("recording");
  progressBar.style.transition = "none";
  progressBar.style.width = "0%";
}

async function upload(blob) {
  const response = await fetch(`/api/upload?kind=${kind}`, {method: "POST", headers: {"Content-Type": "audio/wav"}, body: blob});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Blad zapisu");
  return result.saved;
}

async function recordSingle() {
  const seconds = Math.max(1, Math.min(300, Number(secondsInput.value) || 60));
  setBusy(true);
  recordButton.textContent = "Nagrywanie...";
  try {
    statusBox.textContent = `Nagrywam przez ${seconds} s...`;
    animateProgress(seconds);
    const blob = await capture(seconds);
    resetProgress();
    statusBox.textContent = "Wysylam na komputer...";
    const saved = await upload(blob);
    statusBox.textContent = `Zapisano ${saved.length} plikow: ${saved.join(", ")}`;
    await refreshStats();
  } catch (error) {
    statusBox.textContent = `Blad: ${error.message}`;
  } finally {
    resetProgress();
    setBusy(false);
  }
}

async function recordSeries() {
  const seconds = 2;
  secondsInput.value = seconds;
  setBusy(true);
  stopRequested = false;
  recordButton.textContent = "Stop";
  let savedTotal = 0;
  let sample = 1;
  const session = new CaptureSession();
  try {
    await session.start();
    while (!stopRequested) {
      statusBox.textContent = `Probka ${sample}: nagrywam ${seconds} s...`;
      animateProgress(seconds);
      const blob = await session.record(seconds);
      resetProgress();
      statusBox.textContent = `Probka ${sample}: wysylam...`;
      const saved = await upload(blob);
      savedTotal += saved.length;
      statusBox.textContent = `Zapisano ${savedTotal} plikow. Nastepna probka za chwile...`;
      await refreshStats();
      sample += 1;
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    statusBox.textContent = `Zatrzymano. Zapisano ${savedTotal} plikow z tej serii.`;
  } catch (error) {
    statusBox.textContent = `Blad: ${error.message}. Zapisano ${savedTotal} plikow przed bledem.`;
  } finally {
    await session.stop();
    resetProgress();
    stopRequested = false;
    setBusy(false);
  }
}

recordButton.onclick = async () => {
  if (recording) {
    stopRequested = true;
    recordButton.textContent = "Zatrzymuje...";
    statusBox.textContent = "Koncze biezaca dwusekundowa probke i zatrzymuje serie...";
    return;
  }
  if (isSeriesMode()) await recordSeries();
  else await recordSingle();
};
refreshStats();
</script>
</body>
</html>
"""


def next_index(folder: Path) -> int:
    indices = []
    for path in folder.glob("*.wav"):
        match = INDEX_PATTERN.search(path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def read_pcm_wav(body: bytes) -> tuple[wave._wave_params, bytes]:
    with wave.open(io.BytesIO(body), "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
    if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != SAMPLE_RATE:
        raise ValueError("Oczekiwany format WAV: mono PCM 16-bit, 16000 Hz")
    return params, frames


def write_wav(path: Path, frames: bytes) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(frames)


def save_upload(kind: str, body: bytes) -> list[str]:
    if kind not in DATA_FOLDERS:
        raise ValueError("Nieznany rodzaj nagrania")
    _, frames = read_pcm_wav(body)
    folder = DATA_FOLDERS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    with SAVE_LOCK:
        index = next_index(folder)
        if kind == "positive":
            paths = [folder / f"positive_real_{index:04d}.wav"]
            chunks = [frames]
        else:
            chunk_seconds = HARD_NEGATIVE_CHUNK_SECONDS if kind == "hard_negative" else NEGATIVE_CHUNK_SECONDS
            chunk_size = SAMPLE_RATE * chunk_seconds * 2
            chunks = [frames[start : start + chunk_size] for start in range(0, len(frames), chunk_size)]
            chunks = [chunk for chunk in chunks if len(chunk) >= SAMPLE_RATE * 2]
            prefix = "hard_negative_real" if kind == "hard_negative" else "negative_real"
            paths = [folder / f"{prefix}_{index:04d}_{chunk_index:03d}.wav" for chunk_index in range(len(chunks))]
        for path, chunk in zip(paths, chunks):
            write_wav(path, chunk)
    return [path.name for path in paths]


class RecorderHandler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/stats":
            self.send_json({kind: len(list(folder.glob("*.wav"))) for kind, folder in DATA_FOLDERS.items()})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/upload":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            self.send_json({"error": "NieprawidĹ‚owy rozmiar nagrania"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            saved = save_upload(parse_qs(parsed.query).get("kind", [""])[0], self.rfile.read(content_length))
            self.send_json({"saved": saved})
        except (ValueError, wave.Error, EOFError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pid-file", type=Path, default=BASE_DIR / ".tablet_recorder.pid")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RecorderHandler)
    args.pid_file.write_text(str(os.getpid()), encoding="ascii")
    print(f"Tablet recorder: http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        args.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()


