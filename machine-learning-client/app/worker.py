#!/usr/bin/env python3
"""
Background worker for the ML client.

Reads one pending audio classification task from MongoDB, loads the
corresponding audio bytes from GridFS, extracts features, and classifies
the track as either "vocal" or "electronic" using reference fingerprints.
"""

# pylint: disable=import-error,import-outside-toplevel,invalid-name

import os
import time
from typing import Iterable, Tuple, Any

import numpy as np
from gridfs import GridFS
from pymongo import MongoClient

from app.features import extract_features


def _read_gridfs_audio(gridfs_bucket: GridFS, gridfs_id: Any) -> Tuple[np.ndarray, int]:
    """Read raw audio from GridFS and decode to mono float32 numpy array."""
    import io
    import soundfile as sf

    raw_bytes = gridfs_bucket.get(gridfs_id).read()
    with sf.SoundFile(io.BytesIO(raw_bytes)) as sound_file:
        audio = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sample_rate


def _predict_binary_genre(
    feature_vector: np.ndarray, references: Iterable[dict]
) -> str:
    """Binary ('vocal'/'electronic') prediction using cosine similarity."""
    ref_fps: list[np.ndarray] = []
    labels: list[str] = []

    for doc in references:
        fp = np.asarray(doc.get("fp", []), dtype=float)
        if fp.size == 0:
            continue

        norm = np.linalg.norm(fp)
        if norm == 0:
            continue

        ref_fps.append(fp / norm)
        labels.append(str(doc.get("genre", "")).lower())

    if not ref_fps:
        return "vocal"

    ref_matrix = np.vstack(ref_fps)
    best_index = int(np.argmax(ref_matrix @ feature_vector))
    raw_label = labels[best_index]

    if any(x in raw_label for x in ["vocal", "voice", "singer"]):
        return "vocal"

    if any(x in raw_label for x in ["electronic", "edm", "electro"]):
        return "electronic"

    return "vocal"


def process_one(db, gridfs_bucket: GridFS) -> bool:
    """Process a single pending classification task."""
    task = db.tasks.find_one_and_update(
        {"status": "pending"},
        {"$set": {"status": "processing"}},
    )

    if task is None:
        return False

    try:
        audio, sample_rate = _read_gridfs_audio(gridfs_bucket, task["gridfs_id"])
        feature_vector = extract_features(audio, sample_rate)

        references = db.reference_tracks.find()
        predicted_genre = _predict_binary_genre(feature_vector, references)

        db.results.insert_one(
            {
                "task_id": task["_id"],
                "predicted_genre": predicted_genre,
                "created_at": time.time(),
            }
        )

        db.tasks.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "done", "predicted_genre": predicted_genre}},
        )
        return True

    except Exception as exc:  # pylint: disable=broad-except
        db.tasks.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error_message": str(exc)}},
        )
        return False


def process_loop(db, gridfs_bucket: GridFS, poll_interval: float = 1.0) -> None:
    """Loop forever, processing tasks as they arrive."""
    while True:
        worked = process_one(db, gridfs_bucket)
        if not worked:
            time.sleep(poll_interval)


def create_mongo_client(uri: str) -> MongoClient:
    """Factory for MongoClient (tests patch this)."""
    return MongoClient(uri)


def main() -> None:
    """Entry point for worker container."""
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
    client = create_mongo_client(mongo_uri)
    database = client["ml_audio"]
    gridfs_bucket = GridFS(database)

    process_loop(database, gridfs_bucket)


if __name__ == "__main__":
    main()
