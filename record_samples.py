"""Record real microphone examples for local wake-word fine-tuning."""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000


def save_wav(path: Path, audio: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["positive", "negative"], required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--device", type=int, help="sounddevice input device index")
    args = parser.parse_args()
    try:
        import sounddevice as sd
    except ImportError as error:
        raise SystemExit("Recording requires sounddevice: pip install sounddevice") from error

    if args.device is not None:
        sd.default.device = (args.device, None)
    folder = Path("data/positive_real" if args.kind == "positive" else "data/negative_real")
    count = args.count if args.count is not None else (30 if args.kind == "positive" else 1)
    seconds = args.seconds if args.seconds is not None else (2.0 if args.kind == "positive" else 60.0)
    start_index = len(list(folder.glob("*.wav")))
    print(f"Input device: {sd.query_devices(kind='input')['name']}")
    for index in range(start_index, start_index + count):
        if args.kind == "positive":
            input(f"[{index - start_index + 1}/{count}] Press Enter, then say 'Fikso'...")
        else:
            input(f"Press Enter, then speak normally without saying 'Fikso' for {seconds:.0f} seconds...")
        print("Recording...")
        audio = sd.rec(round(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        if args.kind == "negative":
            chunk_samples = 3 * SAMPLE_RATE
            for chunk_index, start in enumerate(range(0, len(audio), chunk_samples)):
                chunk = audio[start : start + chunk_samples, 0]
                if len(chunk) < SAMPLE_RATE:
                    continue
                path = folder / f"{args.kind}_real_{index:04d}_{chunk_index:03d}.wav"
                save_wav(path, chunk)
                print(f"Saved {path}")
        else:
            path = folder / f"{args.kind}_real_{index:04d}.wav"
            save_wav(path, audio[:, 0])
            print(f"Saved {path}")
        time.sleep(0.25)


if __name__ == "__main__":
    main()
