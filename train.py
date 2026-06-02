"""Train the Polish 'Fikso' wake-word CNN from WAV files in data/."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wakeword import AudioConfig, WakeWordCNN, fit_window, load_wav


def discover_files(data_dir: Path) -> list[tuple[Path, int, str]]:
    required_groups = [("positive_ai", 1), ("hard_negative_ai", 0), ("normal_negative_ai", 0)]
    optional_groups = [("positive_real", 1), ("negative_real", 0)]
    files = []
    for folder, label in required_groups:
        paths = sorted((data_dir / folder).glob("*.wav"))
        if not paths:
            raise FileNotFoundError(f"No WAV files found in {data_dir / folder}")
        files.extend((path, label, folder) for path in paths)
    for folder, label in optional_groups:
        paths = sorted((data_dir / folder).glob("*.wav"))
        files.extend((path, label, folder) for path in paths)
    return files


def stratified_split(files: list[tuple[Path, int, str]], seed: int):
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for group in sorted({item[2] for item in files}):
        items = [item for item in files if item[2] == group]
        rng.shuffle(items)
        n_test = max(1, round(len(items) * 0.15))
        n_validation = max(1, round(len(items) * 0.15))
        test.extend(items[:n_test])
        validation.extend(items[n_test : n_test + n_validation])
        train.extend(items[n_test + n_validation :])
    return train, validation, test


class AudioDataset(Dataset):
    def __init__(self, files, config: AudioConfig, training: bool, seed: int):
        self.items = [(load_wav(path, config.sample_rate), float(label), str(path)) for path, label, _ in files]
        self.negative_audio = [audio for audio, label, _ in self.items if label == 0]
        self.config = config
        self.training = training
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        audio, label, path = self.items[index]
        rng = self.rng if self.training else np.random.default_rng(self.seed + index)
        if label == 0:
            window = np.zeros(self.config.window_samples, dtype=np.float32)
            clips = [] if self.training and rng.random() < 0.25 else [audio] + [self.negative_audio[int(rng.integers(len(self.negative_audio)))] for _ in range(int(rng.integers(1, 4)))]
            for clip in clips:
                start = int(rng.integers(-len(clip) // 3, max(1, len(window) - len(clip) + 1)))
                source_start, target_start = max(0, -start), max(0, start)
                length = min(len(clip) - source_start, len(window) - target_start)
                window[target_start : target_start + length] += clip[source_start : source_start + length]
            window = np.clip(window, -1.0, 1.0)
        else:
            window = fit_window(audio, self.config.window_samples, rng if self.training else None)
        if self.training:
            gain = float(self.rng.uniform(0.65, 1.25))
            noise = self.rng.normal(0.0, self.rng.uniform(0.0, 0.012), len(window)).astype(np.float32)
            window = np.clip(window * gain + noise, -1.0, 1.0)
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.float32), path


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, probabilities, labels, paths = [], [], [], []
    for audio, label, batch_paths in loader:
        logits = model(audio.to(device))
        losses.append(float(loss_fn(logits, label.to(device))))
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(label.tolist())
        paths.extend(batch_paths)
    return float(np.mean(losses)), np.asarray(probabilities), np.asarray(labels), paths


def metrics(probabilities, labels, threshold):
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round((tp + tn) / len(labels), 4),
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, tp + fn), 4),
        "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def choose_threshold(probabilities, labels):
    candidates = np.linspace(0.10, 0.95, 86)
    scored = [metrics(probabilities, labels, value) for value in candidates]
    return max(scored, key=lambda row: (row["f1"], row["precision"], row["threshold"]))["threshold"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fikso_cnn.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--real-repeat", type=int, default=4, help="Repeat real microphone training clips to counterbalance synthetic TTS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = AudioConfig()
    files = discover_files(args.data_dir)
    train_files, validation_files, test_files = stratified_split(files, args.seed)
    real_train_files = [item for item in train_files if item[2].endswith("_real")]
    train_files = train_files + real_train_files * max(0, args.real_repeat - 1)
    print(f"files: train={len(train_files)} validation={len(validation_files)} test={len(test_files)} real_repeat={args.real_repeat}")

    train_loader = DataLoader(AudioDataset(train_files, config, True, args.seed), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(AudioDataset(validation_files, config, False, args.seed), batch_size=args.batch_size)
    test_loader = DataLoader(AudioDataset(test_files, config, False, args.seed), batch_size=args.batch_size)
    model = WakeWordCNN(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state, best_f1, best_threshold = None, -1.0, 0.5
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for audio, label, _ in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(audio.to(args.device)), label.to(args.device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss, probabilities, labels, _ = evaluate(model, validation_loader, loss_fn, args.device)
        threshold = choose_threshold(probabilities, labels)
        validation_metrics = metrics(probabilities, labels, threshold)
        print(f"epoch={epoch:02d} train_loss={np.mean(losses):.4f} val_loss={validation_loss:.4f} val_f1={validation_metrics['f1']:.4f} threshold={threshold:.2f}")
        if validation_metrics["f1"] > best_f1:
            best_f1, best_threshold = validation_metrics["f1"], threshold
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, validation_probabilities, validation_labels, _ = evaluate(model, validation_loader, loss_fn, args.device)
    _, test_probabilities, test_labels, test_paths = evaluate(model, test_loader, loss_fn, args.device)
    report = {
        "seed": args.seed,
        "training_seconds": round(time.perf_counter() - started, 2),
        "files": {"all": len(files), "train": len(train_files), "validation": len(validation_files), "test": len(test_files)},
        "validation": metrics(validation_probabilities, validation_labels, best_threshold),
        "test": metrics(test_probabilities, test_labels, best_threshold),
        "test_errors": [
            {"path": path, "label": int(label), "probability": round(float(probability), 4)}
            for probability, label, path in zip(test_probabilities, test_labels, test_paths)
            if (probability >= best_threshold) != bool(label)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "audio_config": config.to_dict(), "threshold": best_threshold, "metrics": report}, args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
