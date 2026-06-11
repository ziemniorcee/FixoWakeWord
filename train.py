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

POSITIVE_AI_BACKGROUND_SNR_DB = (16.0, 30.0)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-9))


def mix_background(window: np.ndarray, background: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(background) < len(window):
        repeats = int(np.ceil(len(window) / len(background)))
        background = np.tile(background, repeats)
    start = int(rng.integers(0, max(1, len(background) - len(window) + 1)))
    bed = background[start : start + len(window)].astype(np.float32)
    signal_rms = rms(window)
    bed_rms = rms(bed)
    if bed_rms <= 1e-6:
        return window
    snr_db = float(rng.uniform(*POSITIVE_AI_BACKGROUND_SNR_DB))
    target_bed_rms = signal_rms / (10 ** (snr_db / 20.0))
    return np.clip(window + bed * (target_bed_rms / bed_rms), -1.0, 1.0)


def apply_light_reverb(audio: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    wet = float(rng.uniform(0.06, 0.20))
    tail_ms = float(rng.uniform(50.0, 180.0))
    impulse_len = max(1, int(sample_rate * tail_ms / 1000.0))
    impulse = np.zeros(impulse_len, dtype=np.float32)
    impulse[0] = 1.0
    for _ in range(int(rng.integers(2, 6))):
        delay = int(rng.integers(max(1, sample_rate // 300), impulse_len))
        impulse[delay] += float(rng.uniform(0.05, 0.25)) * np.exp(-delay / max(1, impulse_len))
    reverbed = np.convolve(audio, impulse, mode="full")[: len(audio)].astype(np.float32)
    return np.clip((1.0 - wet) * audio + wet * reverbed, -1.0, 1.0)


def discover_files(data_dir: Path) -> list[tuple[Path, int, str]]:
    required_groups = [("positive_real", 1), ("negative_real", 0)]
    optional_groups = [("positive_ai", 1), ("positive_ai_augmented", 1), ("hard_negative_real", 0), ("hard_negative_mined", 0)]
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
        self.items = [(load_wav(path, config.sample_rate), float(label), str(path), group) for path, label, group in files]
        self.negative_audio = [audio for audio, label, _, _ in self.items if label == 0]
        self.config = config
        self.training = training
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        audio, label, path, group = self.items[index]
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
            if self.training and group == "positive_ai" and self.negative_audio and rng.random() < 0.75:
                background = self.negative_audio[int(rng.integers(len(self.negative_audio)))]
                window = mix_background(window, background, rng)
            if self.training and rng.random() < 0.40:
                window = apply_light_reverb(window, self.config.sample_rate, rng)
        if self.training:
            gain_db = float(self.rng.uniform(-10.0, 6.0)) if label == 1 else float(self.rng.uniform(-4.0, 3.0))
            gain = 10.0 ** (gain_db / 20.0)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = AudioConfig()
    files = discover_files(args.data_dir)
    train_files, validation_files, test_files = stratified_split(files, args.seed)
    groups = {group: sum(1 for _, _, item_group in files if item_group == group) for group in sorted({item[2] for item in files})}
    print(f"files: train={len(train_files)} validation={len(validation_files)} test={len(test_files)} groups={groups}")

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
        "files": {"all": len(files), "train": len(train_files), "validation": len(validation_files), "test": len(test_files), "groups": groups},
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
