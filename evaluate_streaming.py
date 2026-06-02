"""Measure streaming detection recall on positive clips embedded in silence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from demo import score_window
from wakeword import DetectionSmoother, iter_stream_windows, load_checkpoint, load_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fikso_cnn.pt"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--required-hits", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("streaming_threshold", checkpoint["threshold"])
    required_hits = args.required_hits if args.required_hits is not None else checkpoint.get("streaming_required_hits", 2)
    silence = np.zeros(model.config.sample_rate, dtype=np.float32)
    detected, total = 0, 0
    misses = []
    for path in sorted((args.data_dir / "positive_ai").glob("*.wav")):
        audio = np.concatenate([silence, load_wav(path, model.config.sample_rate), silence])
        head = DetectionSmoother(threshold, required_hits=required_hits)
        fired = False
        for start, window in iter_stream_windows(audio, model.config):
            timestamp = (start + model.config.window_samples) / model.config.sample_rate
            fired |= head.update(timestamp, score_window(model, window, args.device))
        total += 1
        detected += int(fired)
        if not fired:
            misses.append(str(path))
    report = {
        "threshold": threshold,
        "required_hits": required_hits,
        "positive_clips": total,
        "detected": detected,
        "streaming_recall": round(detected / max(1, total), 4),
        "misses": misses,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
