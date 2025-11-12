# machine-learning-client/app/seed_reference.py
import os
import glob
import random
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from pymongo import MongoClient, errors

# Reuse feature extraction from your project
from .features import extract_features

# ---- Environment variables ----
# FMA_ROOT should point to a folder that contains:
#   - fma_metadata/tracks.csv
#   - fma_small/*/*.mp3
FMA_ROOT = os.getenv("FMA_ROOT", "/app/data/fma")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "musicid")
SEED_LIMIT = int(os.getenv("FMA_SEED_LIMIT", "100"))  # how many mp3s to ingest
RANDOM_SEED = int(os.getenv("FMA_RANDOM_SEED", "42"))

TRACKS_CSV = os.path.join(FMA_ROOT, "fma_metadata", "tracks.csv")
FMA_SMALL_GLOB = os.path.join(FMA_ROOT, "fma_small", "*", "*.mp3")


def _read_tracks_meta(tracks_csv: str) -> pd.DataFrame:
    """
    Read FMA tracks.csv with its multi-index header.
    Returns a DataFrame where columns are a MultiIndex like ('track','genre_top').
    """
    # FMA tracks.csv uses multi-row header; typical read is header=[0, 1]
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    return df


def _to_track_id_from_path(mp3_path: str) -> Optional[int]:
    """
    FMA file names are like '000123.mp3'. Convert basename to int track_id.
    """
    base = os.path.basename(mp3_path)
    name, ext = os.path.splitext(base)
    if not name.isdigit():
        return None
    try:
        return int(name)
    except ValueError:
        return None


def _safe_load_audio(mp3_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from MP3. Prefer soundfile; if it fails, fallback to librosa.
    """
    try:
        y, sr = sf.read(mp3_path)
    except Exception:
        import librosa  # local import to avoid import if not needed
        y, sr = librosa.load(mp3_path, sr=None, mono=True)

    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype("float32", copy=False), sr


def _row_str(df: pd.DataFrame, tid: int, col_group: str, col_name: str) -> Optional[str]:
    """
    Helper: safely fetch a string column like ('track','genre_top') from tracks.csv.
    """
    try:
        val = df.loc[tid, (col_group, col_name)]
        if pd.isna(val):
            return None
        return str(val)
    except Exception:
        return None


def run():
    random.seed(RANDOM_SEED)

    if not os.path.exists(TRACKS_CSV):
        raise FileNotFoundError(
            f"tracks.csv not found at {TRACKS_CSV}. "
            f"Expected FMA_ROOT={FMA_ROOT} with fma_metadata/tracks.csv present."
        )

    mp3_paths = sorted(glob.glob(FMA_SMALL_GLOB))
    if not mp3_paths:
        raise FileNotFoundError(
            f"No MP3 files found under {os.path.dirname(FMA_SMALL_GLOB)}. "
            "Make sure fma_small is extracted."
        )

    # Read meta
    tracks = _read_tracks_meta(TRACKS_CSV)

    # Sample a subset for quicker demo
    if SEED_LIMIT > 0 and SEED_LIMIT < len(mp3_paths):
        mp3_paths = random.sample(mp3_paths, SEED_LIMIT)

    # Connect DB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Create indexes for faster upserts (optional)
    try:
        db.reference_tracks.create_index("track_id", unique=True)
    except errors.PyMongoError:
        pass

    inserted = 0
    for i, mp3 in enumerate(mp3_paths, start=1):
        tid = _to_track_id_from_path(mp3)
        if tid is None:
            # Skip unexpected file names
            continue

        # Fetch metadata from tracks.csv
        genre_top = _row_str(tracks, tid, "track", "genre_top") or "unknown"
        title = _row_str(tracks, tid, "track", "title") or os.path.basename(mp3)
        artist = _row_str(tracks, tid, "artist", "name") or "FMA"

        # Load audio & extract features
        try:
            y, sr = _safe_load_audio(mp3)
            fp = extract_features(y, sr).tolist()
        except Exception as exc:
            # Skip problematic files but continue
            print(f"[WARN] Failed on {mp3}: {exc}")
            continue

        doc = {
            "track_id": tid,
            "title": title,
            "artist": artist,
            "genre": genre_top,
            "fp": fp,
            "source": "FMA_small",
            "license": "See individual FMA track license",
            "created_at": datetime.utcnow(),
        }

        # Upsert by track_id to avoid duplicates
        db.reference_tracks.update_one({"track_id": tid}, {"$set": doc}, upsert=True)
        inserted += 1

        if i % 20 == 0:
            print(f"[seed] processed {i} files, inserted so far: {inserted}")

    print(f"[seed] done. inserted/updated: {inserted}")


if __name__ == "__main__":
    run()
