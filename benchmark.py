"""Benchmark false alarms by streaming all negative WAV files through the detector."""

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
    parser.add_argument("--output", type=Path, default=Path("results/false_alarm_benchmark.json"))
    parser.add_argument("--folders", nargs="+", default=["normal_negative_ai"])
    parser.add_argument("--silence-seconds", type=float, default=1.0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--required-hits", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("streaming_threshold", checkpoint["threshold"])
    required_hits = args.required_hits if args.required_hits is not None else checkpoint.get("streaming_required_hits", 2)
    clips = []
    for folder in args.folders:
        for path in sorted((args.data_dir / folder).glob("*.wav")):
            audio = load_wav(path, model.config.sample_rate)
            clips.extend([audio, np.zeros(round(model.config.sample_rate * args.silence_seconds), dtype=np.float32)])
    audio = np.concatenate(clips)
    seconds = len(audio) / model.config.sample_rate
    detections = []
    head = DetectionSmoother(threshold, required_hits=required_hits)
    for start, window in iter_stream_windows(audio, model.config):
        timestamp = (start + model.config.window_samples) / model.config.sample_rate
        probability = score_window(model, window, args.device)
        if head.update(timestamp, probability):
            detections.append({"timestamp": round(timestamp, 3), "probability": round(probability, 4)})
    report = {
        "negative_audio_minutes": round(seconds / 60.0, 3),
        "folders": args.folders,
        "silence_seconds_between_clips": args.silence_seconds,
        "false_alarms": len(detections),
        "false_alarms_per_hour": round(len(detections) / max(seconds / 3600.0, 1e-9), 3),
        "threshold": threshold,
        "required_hits": required_hits,
        "detections": detections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
