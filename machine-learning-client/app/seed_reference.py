"""
Seed reference fingerprints for two genres:
    - vocal
    - electronic

This script ONLY scans:
    app/reference_audio/vocal/*.wav
    app/reference_audio/electronic/*.wav

Extracts normalized feature vectors and saves them into:
    db.reference_tracks
"""
# pylint: disable=import-error
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
from pymongo import MongoClient
from gridfs import GridFS

from app.features import extract_features

# Base directory: machine-learning-client/app/
BASE_DIR = Path(__file__).resolve().parent

# Reference audio root: machine-learning-client/app/reference_audio/
REF_DIR = BASE_DIR / "reference_audio"

# Only the two genres you want
GENRES = ["vocal", "electronic"]


def load_wav_file(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio file using librosa."""
    audio, sr = librosa.load(path, sr=None)
    return audio, sr


def gather_reference_tracks() -> List[Tuple[str, np.ndarray]]:
    """
    Scan vocal/ and electronic/ folder and extract normalized audio features.
    Returns list of tuples: (genre, feature_vector)
    """
    all_items = []

    for genre in GENRES:
        folder = REF_DIR / genre
        if not folder.exists():
            print(f"⚠ Missing folder: {folder} (skipped)")
            continue

        for wav_file in folder.glob("*.wav"):
            print(f"🎧 Extracting features from: {wav_file}")

            audio, sr = load_wav_file(wav_file)
            fp = extract_features(audio, sr)

            all_items.append((genre, fp))

    return all_items


def seed_reference_tracks(db, fs: GridFS) -> None:
    """Insert fingerprints into MongoDB."""
    db.reference_tracks.delete_many({})
    print("🧹 Cleared old reference tracks.")

    items = gather_reference_tracks()

    for genre, fp in items:
        db.reference_tracks.insert_one(
            {
                "genre": genre,
                "fp": fp.tolist(),
            }
        )
        print(f"✔ Inserted {genre} track (fp length={len(fp)})")

    print("🎉 Finished seeding reference tracks.")


def main():
    """Run from command line."""
    mongo = MongoClient("mongodb://mongodb:27017")
    db = mongo["ml"]
    fs = GridFS(db)
    seed_reference_tracks(db, fs)


if __name__ == "__main__":
    main()

