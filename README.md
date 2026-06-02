# Fikso: Polish wake-word detector

Small end-to-end keyword spotting project for the Polish wake word **"Fikso"**. A compact CNN classifies 1.5-second log-mel spectrogram windows. A streaming head scans audio every 250 ms, ignores silent frames, requires two consecutive positive scores and adds a cooldown to avoid duplicate detections.

## Data

The included `data/` directory contains 800 mono PCM WAV files at 16 kHz:

| Class | Files | Duration | Meaning |
|---|---:|---:|---|
| `positive_ai` | 300 | 5.34 min | "Fikso" alone or in a short phrase |
| `hard_negative_ai` | 300 | 4.26 min | Similar words such as "fiks", "fiko", "fizjo" |
| `normal_negative_ai` | 200 | 4.50 min | Polish phrases without the wake word |

The current data is synthetic speech generated with ElevenLabs voices. This makes the repository immediately reproducible, but it is not a substitute for the assignment target of 100+ recordings per team member and 30 minutes of real background audio. Before the final presentation, add real recordings to the same folders and report results separately for synthetic and real held-out data.

`train.py` discovers WAV files directly instead of trusting manifests. This matters because an interrupted generation run left `data/hard_negative_ai/manifest.csv` incomplete.

## Architecture

The model follows Chapters 22-25: waveform -> 40-bin log-mel spectrogram -> three convolutional blocks -> adaptive pooling -> binary logit. The streaming head connects the window classifier to the real-time use case. An RNN/LSTM from Chapters 32-34 is a reasonable extension, but the CNN is deliberately smaller and easier to train within the one-hour budget.

## Current baseline results

The included checkpoint was trained on CPU in **37.82 seconds** and is **345 KB**. On the fixed held-out split of augmented windows:

| Metric | Value |
|---|---:|
| Accuracy | 0.9583 |
| Precision | 0.9348 |
| Recall | 0.9556 |
| F1 | 0.9451 |

Streaming recall on the 300 synthetic positives embedded in silence is **0.9967**. However, the simulated background benchmark currently produces **199 false alarms over 7.83 minutes** (**1524.10 false alarms/hour**). The hard-negative stress test produces **494 alarms over 17.10 minutes**. This baseline is suitable for demonstrating the full pipeline, but it is not a reliable practical detector yet. The next experiment must collect real long-form background audio and use it for hard-negative mining.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python benchmark.py
python demo.py
```

The microphone demo starts with the conservative `strict` preset because the initial checkpoint only saw synthetic voices:

```powershell
python demo.py --preset strict
python demo.py --preset balanced
python demo.py --preset sensitive
```

If `strict` misses the wake word, collect real microphone samples and retrain:

```powershell
python record_samples.py --kind positive --count 50
python record_samples.py --kind negative --seconds 120
python train.py
python calibrate.py
python demo.py
```

For negative recording, speak normally without saying "Fikso", play room audio or TV audio, and include the phrases that incorrectly triggered the detector. Real WAV files are added to optional `data/positive_real/` and `data/negative_real/` folders automatically discovered by `train.py`.

### Recording samples on an Android tablet over USB

Connect one tablet with USB debugging enabled and run:

```powershell
.\start_tablet_recorder.ps1
```

The script uses `adb reverse`, opens `http://localhost:8765` on the tablet and starts a local collection server. Accept microphone access in the tablet browser when prompted. Positive recordings are saved as two-second WAV files in `data/positive_real/`. Longer negative recordings are split into three-second WAV files in `data/negative_real/`. All files arrive directly on the computer and are ready for `python train.py`.

Stop the foreground server with `Ctrl+C`. If it was started in the background, use:

```powershell
.\stop_tablet_recorder.ps1
```

If the Android SDK is installed elsewhere, start the server manually, configure the reverse port with your `adb`, then open the URL on the tablet:

```powershell
adb reverse tcp:8765 tcp:8765
python tablet_recorder.py
```

`train.py` repeats real microphone training clips four times by default so the synthetic TTS baseline does not dominate them. Change this with `--real-repeat`.

Do not record all real data in one sitting. A useful minimum is:

- three separate positive sessions of 50-100 examples, recorded at different times and distances;
- three separate 10-minute negative sessions: normal speech, TV/music and quiet room or keyboard noise;
- one extra negative session containing phrases that caused false detections during the live demo;
- `python calibrate.py` after every retraining run to save streaming settings into the checkpoint.

`python calibrate.py --min-recall 0.50` creates a stricter demo setting when false alarms matter more than missed wake words.

To retrain, calibrate and export the model into the Android client with one command:

```powershell
.\update_android_model.cmd
```

The script copies the checkpoint to `C:\Dev\untitled\service-assistant\client`, regenerates the native `fikso_cnn.bin` asset and synchronizes the calibrated streaming threshold and required hit count. Rebuild and install the Android application after it finishes. Use `.\update_android_model.cmd -SkipTraining` to re-export the current checkpoint without training it again.

To verify streaming without a microphone:

```powershell
python demo.py --file data\positive_ai\positive_00000.wav --verbose
```

The trained checkpoint is written to `checkpoints/fikso_cnn.pt` and metrics to `results/`. Training uses a fixed seed and stratified 70/15/15 splits for the three source folders.

`python benchmark.py` measures background-like negatives, inserting one second of silence between the available short TTS clips. Run the intentionally harder phonetic stress test separately:

```powershell
python benchmark.py --folders hard_negative_ai normal_negative_ai --output results\stress_test.json
```

## What to report honestly

- Main classification metrics: precision, recall and F1 on the held-out test split.
- Streaming metric: false alarms per hour from `python benchmark.py`.
- Error analysis: inspect `results/metrics.json` and benchmark detections.
- Known risk: synthetic voices and short isolated negative clips are easier than TV, music and live room audio.
- Current failure: the clip classifier scores many sliding windows from synthetic background phrases too highly, so false-alarm rate is not acceptable yet.
- Final data collection: add team recordings and at least 30 minutes of long-form background audio before making claims about practical reliability.

## Repository map

| File | Purpose |
|---|---|
| `train.py` | Reproducible raw WAV -> checkpoint training |
| `demo.py` | Laptop-mic real-time demo and WAV streaming mode |
| `benchmark.py` | False-alarm benchmark on negative audio |
| `evaluate_streaming.py` | Streaming recall on positive clips embedded in silence |
| `record_samples.py` | Record real positives and background audio from a microphone |
| `calibrate.py` | Select and save streaming threshold from real recordings |
| `wakeword.py` | WAV loading, log-mel frontend, CNN and streaming head |
| `generate_fixo_tts_dataset.py` | Optional synthetic-data generator |
| `REPORT.md` | Ready-to-expand report structure |
| `SLIDES.md` | 10-minute presentation outline |
