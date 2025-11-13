#!/usr/bin/env python3
"""
Background worker for the ML client.

Reads one pending audio classification task from MongoDB, loads the
corresponding audio bytes from GridFS, extracts features, and classifies
the track as either "vocal" or "electronic" using the reference fingerprints
stored in the database.
"""
# pylint: disable=import-error
import os
import time
from typing import Iterable, Tuple, Any

import numpy as np
from gridfs import GridFS
from pymongo import MongoClient

from app.features import extract_features


def _read_gridfs_audio(gridfs_bucket: GridFS, gridfs_id: Any) -> Tuple[np.ndarray, int]:
    """
    Read raw audio data from GridFS and decode to mono float32 numpy array.

    Parameters
    ----------
    gridfs_bucket:
        GridFS bucket used to fetch the audio blob.
    gridfs_id:
        Identifier of the stored audio file in GridFS.

    Returns
    -------
    audio : np.ndarray
        Mono audio signal as float32.
    sr : int
        Sampling rate of the audio.
    """
    import io
    import soundfile as sf  # local import to keep worker importable without SF

    raw_bytes = gridfs_bucket.get(gridfs_id).read()
    with sf.SoundFile(io.BytesIO(raw_bytes)) as sound_file:
        audio = sound_file.read(dtype="float32")
        sr = sound_file.samplerate

    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr


def _predict_binary_genre(
    feature_vector: np.ndarray, references: Iterable[dict]
) -> str:
    """
    Predict a binary genre label ("vocal" or "electronic") for the given feature vector.

    The function looks at all reference fingerprints in the database, finds
    the most similar one (cosine similarity), and then maps its genre into
    the binary space:

    - Any label containing "vocal", "voice", "singer" -> "vocal"
    - Any label containing "electronic", "edm", "electro" -> "electronic"
    - Otherwise falls back to "vocal" (defensive default)

    Parameters
    ----------
    feature_vector:
        Normalized feature vector for the current track.
    references:
        Iterable of documents from the `reference_tracks` collection.

    Returns
    -------
    genre : str
        Either "vocal" or "electronic".
    """
    ref_fps: list[np.ndarray] = []
    labels: list[str] = []

    for doc in references:
        fp = np.asarray(doc.get("fp", []), dtype=float)
        if fp.size == 0:
            continue

        norm = np.linalg.norm(fp)
        if norm == 0:
            continue

        fp = fp / norm
        ref_fps.append(fp)
        labels.append(str(doc.get("genre", "")).lower())

    # 如果数据库里不能匹配到合适 reference，给一个保底结果，避免崩掉
    if not ref_fps:
        return "vocal"

    ref_matrix = np.vstack(ref_fps)    
    similarities = ref_matrix @ feature_vector
    best_index = int(np.argmax(similarities))
    raw_label = labels[best_index]

    if "vocal" in raw_label or "voice" in raw_label or "singer" in raw_label:
        return "vocal"

    if "electronic" in raw_label or "edm" in raw_label or "electro" in raw_label:
        return "electronic"

    # 正常情况下 seed_reference 只会写入 vocal / electronic，
    return "vocal"


def process_one(db, gridfs_bucket: GridFS) -> bool:
    """
    Process a single pending classification task.

    Workflow:
    1. 从 db.tasks 取出一个 status="pending" 的任务并置为 "processing"
    2. 从 GridFS 读出对应音频
    3. 提取音频特征向量
    4. 从 db.reference_tracks 取出所有参考指纹，预测 "vocal"/"electronic"
    5. 把结果写入 db.results，同时更新 task 状态为 "done" 和 predicted_genre

    Parameters
    ----------
    db:
        Pymongo database handle.
    gridfs_bucket:
        GridFS bucket used to fetch the audio.

    Returns
    -------
    worked : bool
        True if a task was processed, False if there was no pending task.
    """
    task = db.tasks.find_one_and_update(
        {"status": "pending"},
        {"$set": {"status": "processing"}},
    )

    # 没有任务就直接返回 False
    if task is None:
        return False

    try:
        audio, sample_rate = _read_gridfs_audio(gridfs_bucket, task["gridfs_id"])
        feature_vector = extract_features(audio, sample_rate)

        references = db.reference_tracks.find()
        predicted_genre = _predict_binary_genre(feature_vector, references)

        result_doc = {
            "task_id": task["_id"],
            "predicted_genre": predicted_genre,
            "created_at": time.time(),
        }
        db.results.insert_one(result_doc)

        db.tasks.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "status": "done",
                    "predicted_genre": predicted_genre,
                }
            },
        )
        return True
    except Exception as exc:  # pylint: disable=broad-except
        # 出错时把任务标记成 error，方便你在 Web UI 或 Mongo 里排查
        db.tasks.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error_message": str(exc)}},
        )
        return False


def process_loop(db, gridfs_bucket: GridFS, poll_interval: float = 1.0) -> None:
    """
    Loop forever, processing tasks as they arrive.

    When there is no pending task, sleep for `poll_interval` seconds.
    """
    while True:
        worked = process_one(db, gridfs_bucket)
        if not worked:
            time.sleep(poll_interval)


def create_mongo_client(uri: str) -> MongoClient:
    """
    Helper to create a MongoClient, kept in a separate function
    so that it is easy to stub or patch in tests.
    """
    return MongoClient(uri)


def main() -> None:
    """
    Entry point when running the worker as a standalone container.
    """
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
    client = create_mongo_client(mongo_uri)
    database = client["ml_audio"]
    gridfs_bucket = GridFS(database)

    process_loop(database, gridfs_bucket)


if __name__ == "__main__":
    main()

