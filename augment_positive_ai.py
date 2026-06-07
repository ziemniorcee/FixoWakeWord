"""Create realistic positive wake-word windows from synthetic "Fikso" clips."""

from __future__ import annotations

import argparse
import csv
import re
import wave
from pathlib import Path

import numpy as np

from wakeword import AudioConfig, load_wav


WORD_PATTERN = re.compile(r"[^a-z]+")


def normalize_text(text: str) -> str:
    return WORD_PATTERN.sub("", text.casefold())


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def trim_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.012, padding_ms: int = 40) -> np.ndarray:
    frame = max(1, int(sample_rate * 0.02))
    if len(audio) <= frame:
        return audio.copy()
    energy = np.array([np.sqrt(np.mean(audio[start : start + frame] ** 2)) for start in range(0, len(audio), frame)])
    active = np.flatnonzero(energy > threshold)
    if len(active) == 0:
        return audio.copy()
    padding = int(sample_rate * padding_ms / 1000)
    start = max(0, int(active[0]) * frame - padding)
    end = min(len(audio), (int(active[-1]) + 1) * frame + padding)
    return audio[start:end].copy()


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-9))


def scale_to_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    return audio * (target_rms / max(rms(audio), 1e-6))


def apply_reverb(audio: np.ndarray, sample_rate: int, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    wet = float(rng.uniform(0.08, 0.28))
    tail_ms = float(rng.uniform(60.0, 220.0))
    impulse_len = max(1, int(sample_rate * tail_ms / 1000.0))
    impulse = np.zeros(impulse_len, dtype=np.float32)
    impulse[0] = 1.0
    echo_count = int(rng.integers(3, 8))
    for _ in range(echo_count):
        delay = int(rng.integers(max(1, sample_rate // 250), impulse_len))
        decay = float(rng.uniform(0.08, 0.35)) * np.exp(-delay / max(1, impulse_len))
        impulse[delay] += decay
    reverbed = np.convolve(audio, impulse, mode="full")[: len(audio)].astype(np.float32)
    mixed = (1.0 - wet) * audio + wet * reverbed
    return np.clip(mixed, -1.0, 1.0), wet, tail_ms


def pick_background(backgrounds: list[np.ndarray], samples: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    if not backgrounds or rng.random() < 0.25:
        return np.zeros(samples, dtype=np.float32), "silence"
    background = backgrounds[int(rng.integers(len(backgrounds)))]
    if len(background) < samples:
        background = np.tile(background, int(np.ceil(samples / len(background))))
    start = int(rng.integers(0, max(1, len(background) - samples + 1)))
    bed = background[start : start + samples].astype(np.float32)
    return bed, "negative_real"


def mix_at_snr(background: np.ndarray, word: np.ndarray, snr_db: float) -> np.ndarray:
    if rms(background) <= 1e-6:
        return background
    word_rms = rms(word)
    target_background_rms = word_rms / (10.0 ** (snr_db / 20.0))
    return scale_to_rms(background, target_background_rms).astype(np.float32)


def discover_sources(input_dir: Path) -> list[Path]:
    manifest_path = input_dir / "manifest.csv"
    if not manifest_path.exists():
        return sorted(input_dir.glob("*.wav"))

    sources = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if normalize_text(row.get("text", "")) != "fikso":
                continue
            path = Path(row["path"])
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                sources.append(path)
    return sorted(sources)


def make_augmented_window(
    word: np.ndarray,
    backgrounds: list[np.ndarray],
    config: AudioConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float | str]]:
    samples = config.window_samples
    word = trim_silence(word, config.sample_rate)
    if len(word) > int(samples * 0.85):
        start = (len(word) - int(samples * 0.85)) // 2
        word = word[start : start + int(samples * 0.85)]

    pre_ms = float(rng.uniform(100.0, 500.0))
    post_ms = float(rng.uniform(100.0, 500.0))
    offset_ms = float(rng.uniform(-120.0, 120.0))
    start = int(config.sample_rate * (pre_ms + offset_ms) / 1000.0)
    min_start = int(config.sample_rate * 0.10)
    max_start = samples - len(word) - int(config.sample_rate * post_ms / 1000.0)
    start = max(min_start, min(max_start, start))
    start = max(0, min(samples - len(word), start))

    window, background_kind = pick_background(backgrounds, samples, rng)
    gain_db = float(rng.uniform(-10.0, 6.0))
    target_word = np.clip(word * (10.0 ** (gain_db / 20.0)), -1.0, 1.0)
    reverb_applied = rng.random() < 0.40
    reverb_wet, reverb_tail_ms = 0.0, 0.0
    if reverb_applied:
        target_word, reverb_wet, reverb_tail_ms = apply_reverb(target_word, config.sample_rate, rng)
    snr_db = float(rng.uniform(0.0, 20.0))
    if background_kind != "silence":
        window = mix_at_snr(window, target_word, snr_db)
    window[start : start + len(target_word)] += target_word
    peak = float(np.max(np.abs(window)))
    if peak > 0.98:
        window = window * (0.98 / peak)
    return window.astype(np.float32), {
        "pre_ms": round(pre_ms, 2),
        "post_ms": round(post_ms, 2),
        "offset_ms": round(offset_ms, 2),
        "gain_db": round(gain_db, 2),
        "snr_db": round(snr_db, 2) if background_kind != "silence" else "",
        "reverb_applied": str(reverb_applied).lower(),
        "reverb_wet": round(reverb_wet, 4),
        "reverb_tail_ms": round(reverb_tail_ms, 2),
        "background": background_kind,
        "word_start_ms": round(start * 1000.0 / config.sample_rate, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/positive_ai"))
    parser.add_argument("--output", type=Path, default=Path("data/positive_ai_augmented"))
    parser.add_argument("--background-folders", type=Path, nargs="+", default=[Path("data/negative_real"), Path("data/hard_negative_real")])
    parser.add_argument("--variants-per-source", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = AudioConfig()
    rng = np.random.default_rng(args.seed)
    sources = discover_sources(args.input)
    if not sources:
        raise SystemExit(f"No standalone Fikso WAV files found in {args.input}")

    background_paths = []
    for folder in args.background_folders:
        background_paths.extend(sorted(folder.glob("*.wav")))
    backgrounds = [load_wav(path, config.sample_rate) for path in background_paths]

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "label", "kind", "source", "variant", "pre_ms", "post_ms", "offset_ms", "gain_db", "snr_db", "reverb_applied", "reverb_wet", "reverb_tail_ms", "background", "word_start_ms"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        index = 0
        for source in sources:
            word = load_wav(source, config.sample_rate)
            for variant in range(args.variants_per_source):
                window, params = make_augmented_window(word, backgrounds, config, rng)
                output_path = args.output / f"positive_ai_augmented_{index:05d}.wav"
                save_wav(output_path, window, config.sample_rate)
                writer.writerow({
                    "path": str(output_path),
                    "label": 1,
                    "kind": "positive_ai_augmented",
                    "source": str(source),
                    "variant": variant,
                    **params,
                })
                index += 1
    print(f"created {index} augmented positives in {args.output}")


if __name__ == "__main__":
    main()
