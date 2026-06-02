import os
import csv
import time
import wave
import random
import argparse
from pathlib import Path

import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.elevenlabs.io/v1"

POLISH_VOICES = [
    {"voice_id": "V5GZ9rfeV9jjKZE5NkT7", "name": "polish_voice_1"},
    {"voice_id": "fEfGdiGJK4l3imI70mtC", "name": "polish_voice_2"},
    {"voice_id": "OOTZSkkPGHD1csczSCmT", "name": "polish_voice_3"},
    {"voice_id": "d4Z5Fvjohw3zxGpV8XUV", "name": "polish_voice_4"},
    {"voice_id": "JWUOwsYG4XgR9Od3eeon", "name": "polish_voice_5"},
    {"voice_id": "g8ZOdhoD9R6eYKPTjKbE", "name": "polish_voice_6"},
    {"voice_id": "pb8c0r30vHTjqkuyCUIQ", "name": "polish_voice_7"},
    {"voice_id": "ELipIT4gf7mqOUIdVqNN", "name": "polish_voice_8"},
]


POSITIVE_TEXTS = [
    "fikso",
    "Fikso",
    "Fikso!",
    "Hej Fikso",
    "Fikso, pomóż",
    "Fikso, start",
    "Fikso, pokaż błąd",
]


# Trudne negatywy: brzmią podobnie, ale NIE są wake wordem.
HARD_NEGATIVE_TEXTS = [
    "fiks",
    "fizo",
    "fiko",
    "fix",
    "fizjo",
    "fikcja",
    "fiskus",
    "pixel",
    "wszystko",
    "szybko",
    "filtr",
    "firma",
    "focus",
    "fiasko",
]


# Łatwe negatywy: normalne krótkie frazy serwisowe bez wake worda.
NORMAL_NEGATIVE_TEXTS = [
    "pokaż błąd",
    "co oznacza ten kod",
    "otwórz instrukcję",
    "sprawdź baterię",
    "podaj numer błędu",
    "nie działa podnoszenie",
    "wózek nie jedzie",
    "sprawdź czujnik",
    "potrzebuję pomocy",
    "uruchom diagnostykę",
    "czy to jest bezpieczne",
    "gdzie jest bezpiecznik",
    "pokaż schemat",
    "jaki to parametr",
]


VOICE_SETTINGS_POOL = [
    {"stability": 0.35, "similarity_boost": 0.70, "style": 0.00, "use_speaker_boost": True, "speed": 0.90},
    {"stability": 0.45, "similarity_boost": 0.75, "style": 0.05, "use_speaker_boost": True, "speed": 1.00},
    {"stability": 0.55, "similarity_boost": 0.80, "style": 0.10, "use_speaker_boost": True, "speed": 1.10},
    {"stability": 0.65, "similarity_boost": 0.85, "style": 0.15, "use_speaker_boost": True, "speed": 1.20},
]


def get_api_key() -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Brak ELEVENLABS_API_KEY w zmiennych środowiskowych.")
    return api_key


def list_voices(api_key: str) -> list[dict]:
    response = requests.get(
        f"{API_BASE}/voices",
        headers={"xi-api-key": api_key},
        timeout=30,
    )

    if not response.ok:
        print("ElevenLabs /voices error")
        print("Status:", response.status_code)
        print("Response:", response.text)
        response.raise_for_status()

    data = response.json()

    if "voices" not in data:
        print("Unexpected response:", data)
        raise RuntimeError("Brak pola 'voices' w odpowiedzi ElevenLabs.")

    return data["voices"]


def save_pcm16_as_wav(pcm_bytes: bytes, output_path: Path, sample_rate: int = 16000):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def generate_tts_pcm(
    api_key: str,
    voice_id: str,
    text: str,
    model_id: str,
    voice_settings: dict,
) -> bytes:
    url = f"{API_BASE}/text-to-speech/{voice_id}?output_format=pcm_16000"

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": voice_settings,
    }

    response = requests.post(
        url,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        },
        json=payload,
        timeout=60,
    )

    if response.status_code == 429:
        raise RuntimeError("Rate limit albo quota exceeded.")

    response.raise_for_status()
    return response.content


def pick_text(label: str) -> str:
    if label == "positive":
        return random.choice(POSITIVE_TEXTS)

    if label == "hard_negative":
        return random.choice(HARD_NEGATIVE_TEXTS)

    if label == "normal_negative":
        return random.choice(NORMAL_NEGATIVE_TEXTS)

    raise ValueError(f"Nieznany label: {label}")


def label_to_numeric(label: str) -> int:
    if label == "positive":
        return 1

    if label in ["hard_negative", "normal_negative"]:
        return 0

    raise ValueError(f"Nieznany label: {label}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        choices=["positive", "hard_negative", "normal_negative"],
        required=True,
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--voices", type=int, default=8)
    parser.add_argument("--model", type=str, default="eleven_multilingual_v2")
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    api_key = get_api_key()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_voices = POLISH_VOICES[: args.voices]

    if not selected_voices:
        raise RuntimeError("Nie znaleziono głosów na koncie ElevenLabs.")

    numeric_label = label_to_numeric(args.label)

    print(f"Tryb: {args.label}")
    print(f"Etykieta numeryczna: {numeric_label}")
    print(f"Folder: {out_dir}")
    print(f"Liczba głosów: {len(selected_voices)}")

    manifest_path = out_dir / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "kind",
                "text",
                "voice_id",
                "voice_name",
                "model_id",
                "stability",
                "similarity_boost",
                "style",
                "speed",
            ],
        )
        writer.writeheader()

        for i in tqdm(range(args.count), desc=f"Generating {args.label}"):
            voice = random.choice(selected_voices)
            text = pick_text(args.label)
            settings = random.choice(VOICE_SETTINGS_POOL)

            filename = f"{args.label}_{i:05d}.wav"
            output_path = out_dir / filename

            try:
                pcm = generate_tts_pcm(
                    api_key=api_key,
                    voice_id=voice["voice_id"],
                    text=text,
                    model_id=args.model,
                    voice_settings=settings,
                )

                save_pcm16_as_wav(pcm, output_path, sample_rate=16000)

                writer.writerow({
                    "path": str(output_path),
                    "label": numeric_label,
                    "kind": args.label,
                    "text": text,
                    "voice_id": voice["voice_id"],
                    "voice_name": voice.get("name", ""),
                    "model_id": args.model,
                    "stability": settings["stability"],
                    "similarity_boost": settings["similarity_boost"],
                    "style": settings["style"],
                    "speed": settings["speed"],
                })

                time.sleep(args.sleep)

            except Exception as e:
                print(f"\nBłąd przy próbce {i}: {e}")
                time.sleep(3)

    print(f"Gotowe: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
