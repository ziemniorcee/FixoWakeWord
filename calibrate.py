"""Choose streaming detection settings from real microphone recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from demo import score_window
from wakeword import DetectionSmoother, iter_stream_windows, load_checkpoint, load_wav


def window_scores(model, audio, device):
    return [
        score_window(model, window, device)
        for _, window in iter_stream_windows(audio, model.config)
    ]


def count_detections(scores, threshold, required_hits, hop_seconds):
    head = DetectionSmoother(threshold, required_hits=required_hits)
    return sum(head.update((index + 1) * hop_seconds, score) for index, score in enumerate(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fikso_cnn.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/calibration.json"))
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    positives = sorted((args.data_dir / "positive_real").glob("*.wav"))
    negatives = sorted((args.data_dir / "negative_real").glob("*.wav"))
    negatives += sorted((args.data_dir / "hard_negative_real").glob("*.wav"))
    if not positives or not negatives:
        raise SystemExit("Calibration requires WAV files in data/positive_real and at least one real negative folder")

    silence = np.zeros(model.config.sample_rate, dtype=np.float32)
    positive_scores = [
        window_scores(model, np.concatenate([silence, load_wav(path), silence]), args.device)
        for path in positives
    ]
    # Recorder chunks are contiguous slices from background sessions.
    negative_audio = np.concatenate([load_wav(path) for path in negatives])
    negative_scores = window_scores(model, negative_audio, args.device)
    negative_hours = len(negative_audio) / model.config.sample_rate / 3600.0
    candidates = []
    for required_hits in (2, 3, 4):
        for threshold in np.linspace(0.50, 0.995, 100):
            detected = sum(count_detections(scores, threshold, required_hits, model.config.hop_seconds) > 0 for scores in positive_scores)
            recall = detected / len(positive_scores)
            false_alarms = count_detections(negative_scores, threshold, required_hits, model.config.hop_seconds)
            candidates.append({
                "threshold": round(float(threshold), 3),
                "required_hits": required_hits,
                "streaming_recall": round(recall, 4),
                "false_alarms": false_alarms,
                "false_alarms_per_hour": round(false_alarms / max(negative_hours, 1e-9), 3),
            })
    eligible = [row for row in candidates if row["streaming_recall"] >= args.min_recall]
    best = min(eligible or candidates, key=lambda row: (row["false_alarms_per_hour"], -row["streaming_recall"], row["threshold"]))
    checkpoint["streaming_threshold"] = best["threshold"]
    checkpoint["streaming_required_hits"] = best["required_hits"]
    torch.save(checkpoint, args.checkpoint)
    report = {
        "positive_real_clips": len(positives),
        "negative_real_clips": len(negatives),
        "negative_real_minutes": round(negative_hours * 60.0, 3),
        "minimum_requested_recall": args.min_recall,
        "selected": best,
        "warning": None if eligible else "No candidate reached the requested recall; more varied real training data is needed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
