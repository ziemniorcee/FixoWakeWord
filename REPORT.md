# Fikso wake-word detector: report draft

## 1. Motivation

The goal is a small Polish keyword-spotting system that detects "Fikso" in streaming microphone audio. The practical constraint is more important than leaderboard accuracy: training should finish in under one hour and the final model should run locally on a laptop.

## 2. Related course material

The main model uses Chapters 22-25: convolution over a time-frequency representation. Audio is converted to a 40-bin log-mel spectrogram, which exposes local frequency-time patterns such as syllables. Chapters 32-34 motivate a possible recurrent extension, but a CNN is sufficient for the first working baseline and cheaper to train.

## 3. Data

The initial dataset contains 800 synthetic ElevenLabs WAV files: 300 positive examples, 300 phonetically difficult negatives and 200 normal Polish negative phrases. All recordings are mono 16-bit PCM at 16 kHz. The total duration is 14.10 minutes. The final experiment should add real team voices and at least 30 minutes of background audio.

## 4. Model and streaming logic

The model receives 1.5-second audio windows. It computes normalized log-mel features and applies three convolutional blocks with 12, 24 and 32 channels. Adaptive pooling preserves a small time-frequency map before the binary classification head. During streaming, windows advance every 250 ms. A lightweight RMS gate ignores silent frames. A wake event fires only after two consecutive scores exceed the validation-set threshold, followed by a 1.5-second cooldown.

## 5. Training

`train.py` uses a fixed random seed, stratified 70/15/15 file-level splits, Adam, binary cross-entropy and lightweight gain/noise augmentation. Add the final runtime, threshold and metrics from `results/metrics.json`.

## 6. Results

The final reported CPU run finished in 37.82 seconds. The checkpoint is 345 KB. On the held-out augmented-window test split, accuracy was 0.9583, precision 0.9348, recall 0.9556 and F1 0.9451. Streaming recall on 300 synthetic positives embedded in silence was 0.9967.

The false-alarm result is the main limitation: the simulated background benchmark produced 199 detections over 7.83 minutes, or 1524.10 false alarms per hour. The intentionally adversarial benchmark including hard negatives produced 494 detections over 17.10 minutes. The current baseline demonstrates the software pipeline but does not meet a practical wake-word reliability target.

## 7. Error analysis

Use `test_errors` from `results/metrics.json` and `detections` from `results/false_alarm_benchmark.json`. Listen to false positives and group them into similar-sounding words, noise, speaker variation and window-boundary failures. Keep the background benchmark separate from the deliberately adversarial hard-negative stress test. The present experiment already shows a mismatch between strong held-out window classification and poor sliding-window FAR. The most valuable next step is hard-negative mining from real TV, music and room audio.

## 8. Lessons learned

The useful engineering lesson is that offline clip accuracy is not enough. Streaming hysteresis, cooldown behavior and a long-form background benchmark directly affect whether a wake-word demo feels reliable.
