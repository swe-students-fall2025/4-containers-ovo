"""Batch extractor: scan labeled subfolders and write one CSV.

Place your audio organized like:
    data/rock_vs_hiphop_database/hiphop/*.mp3
    data/rock_vs_hiphop_database/rock/*.mp3

Then run:
    python -m machine_learning_client.app.batch_extract --parent "...\\..\\data\\rock_vs_hiphop_database" --output "...\\..\\data\\features_all.csv"

Or from within the `machine-learning-client/app` folder:
    python batch_extract.py --parent "../../data/rock_vs_hiphop_database" --output "../../data/features_all.csv"

The script uses each immediate subfolder name as the label.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable

import librosa

# Handle both direct script execution and module import
try:
    from .features import extract_features_audio
except ImportError:
    # When running script directly, add parent to path
    sys.path.insert(0, os.path.dirname(__file__))
    from features import extract_features_audio


def iter_audio_files(path: str, exts: Iterable[str] = (".mp3", ".wav", ".flac")):
    for root, _, files in os.walk(path):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in exts):
                yield os.path.join(root, fname)


def process_parent(parent_dir: str, output_csv: str, sr: int | None = None, exts: Iterable[str] = (".mp3",)):
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

    subfolders = [d for d in sorted(os.listdir(parent_dir)) if os.path.isdir(os.path.join(parent_dir, d))]
    if not subfolders:
        raise SystemExit(f"No subfolders found in {parent_dir}; expected labeled folders like 'hiphop' and 'rock'.")

    total = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for label in subfolders:
            folder = os.path.join(parent_dir, label)
            for path in iter_audio_files(folder, exts=exts):
                total += 1
                try:
                    y, sample_rate = librosa.load(path, sr=sr, mono=True)
                    vec = extract_features_audio(y, sample_rate)
                    row = [float(x) for x in vec.tolist()]
                    row.append(label)
                    row.append(os.path.basename(path))
                    writer.writerow(row)
                except Exception as exc:
                    print(f"Warning: failed to process '{path}': {exc}")

    print(f"Wrote features for ~{total} files to {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch extract audio features from labeled subfolders")
    p.add_argument("--parent", required=True, help="Parent folder containing labeled subfolders")
    p.add_argument("--output", required=True, help="Output CSV file")
    p.add_argument("--sr", type=int, default=None, help="Resample rate (None keeps original)")
    p.add_argument("--exts", nargs="*", default=[".mp3", ".wav"], help="Audio extensions to include")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    process_parent(args.parent, args.output, args.sr, exts=tuple(args.exts))
