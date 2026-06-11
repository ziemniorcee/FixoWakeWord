"""Mine high-scoring background windows to retrain as hard negatives."""

from __future__ import annotations

import argparse
import heapq
import wave
from pathlib import Path

import numpy as np
import torch

from demo import score_window
from wakeword import iter_stream_windows, load_checkpoint, load_wav


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    samples = np.clip(audio, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fikso_cnn.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hard_negative_mined"))
    parser.add_argument("--folders", nargs="+", default=["negative_real", "hard_negative_real"])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-per-file", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, _ = load_checkpoint(args.checkpoint, args.device)
    best: list[tuple[float, int, str, int, np.ndarray]] = []
    counter = 0
    total_windows = 0

    for folder in args.folders:
        for path in sorted((args.data_dir / folder).glob("*.wav")):
            audio = load_wav(path, model.config.sample_rate)
            file_hits: list[tuple[float, int, str, int, np.ndarray]] = []
            for start, window in iter_stream_windows(audio, model.config):
                total_windows += 1
                probability = score_window(model, window, args.device)
                if probability < args.threshold:
                    continue
                counter += 1
                item = (probability, counter, str(path), start, window.copy())
                if len(file_hits) < args.max_per_file:
                    heapq.heappush(file_hits, item)
                elif probability > file_hits[0][0]:
                    heapq.heapreplace(file_hits, item)
            for item in file_hits:
                if len(best) < args.max_samples:
                    heapq.heappush(best, item)
                elif item[0] > best[0][0]:
                    heapq.heapreplace(best, item)

    selected = sorted(best, reverse=True)
    for index, (probability, _, source, start, window) in enumerate(selected):
        out = args.output_dir / f"hard_negative_mined_{index:04d}_score_{probability:.3f}.wav"
        write_wav(out, window, model.config.sample_rate)

    print(
        f"scanned_windows={total_windows} selected={len(selected)} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
