"""
Command-line utility to extract audio features for a folder of audio files
and write them to a CSV for training the rock vs hip-hop classifier.

Usage (powershell):
    python extract_features.py --input-dir ../data/my_mp3s --output features.csv --label hiphop

Dependencies:
    pip install librosa numpy

Note: librosa may require `ffmpeg` / `libsndfile` for MP3 support on some systems.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable

import librosa
import numpy as np

from .features import extract_features_audio


def iter_audio_files(path: str, exts: Iterable[str] = (".mp3", ".wav", ".flac")):
    for root, _, files in os.walk(path):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in exts):
                yield os.path.join(root, fname)


def process_folder(input_dir: str, output_csv: str, label: str = "hiphop", sr: int | None = None):
    header = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "tempo",
        "valence",
        "label",
        "filename",
    ]

    total = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for path in iter_audio_files(input_dir):
            total += 1
            try:
                y, sample_rate = librosa.load(path, sr=sr, mono=True)
                vec = extract_features_audio(y, sample_rate)
                row = [float(x) for x in vec.tolist()]  # 8 features
                row.append(label)
                row.append(os.path.basename(path))
                writer.writerow(row)
            except Exception as exc:  # keep going on single-file failures
                print(f"Warning: failed to process '{path}': {exc}")

    print(f"Wrote features for ~{total} files to {output_csv}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract audio features to CSV")
    p.add_argument("--input-dir", required=True, help="Folder containing audio files (MP3/WAV/FLAC)")
    p.add_argument("--output", required=True, help="Destination CSV file path")
    p.add_argument("--label", default="hiphop", help="Label to write for every row")
    p.add_argument("--sr", type=int, default=None, help="Resample rate (None keeps original)")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    process_folder(args.input_dir, args.output, args.label, args.sr)
