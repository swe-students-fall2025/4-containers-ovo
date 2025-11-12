import os
import io
import time
from datetime import datetime
from typing import List, Dict
import numpy as np
import soundfile as sf
from pymongo import MongoClient
from gridfs import GridFS
from .features import extract_features, cosine_sim

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "musicid")
BUCKET = os.getenv("GRIDFS_BUCKET", "audio")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SEC", "1.0"))

def _connect():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = GridFS(db, collection=BUCKET)
    return db, fs

def _load_reference(db) -> List[Dict]:
    return list(db.reference_tracks.find({}, {"_id": 0, "title": 1, "genre": 1, "fp": 1}))

def _read_gridfs_audio(fs: GridFS, file_id):
    blob = fs.get(file_id).read()
    buff = io.BytesIO(blob)
    try:
        y, sr = sf.read(buff)
    except Exception:
        import librosa
        buff.seek(0)
        y, sr = librosa.load(buff, sr=None, mono=True)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype("float32", copy=False), sr

def _predict_genre(fp_test: np.ndarray, references: List[Dict]):
    scores = {}
    for ref in references:
        s = cosine_sim(fp_test, np.asarray(ref["fp"], dtype=float))
        g = ref.get("genre") or "unknown"
        scores[g] = scores.get(g, 0.0) + float(s)
    if not scores:
        return "unknown", 0.0
    genre = max(scores.items(), key=lambda kv: kv[1])[0]
    confidence = scores[genre] / (sum(scores.values()) + 1e-8)
    return genre, float(confidence)

def process_one(db, fs) -> bool:
    task = db.tasks.find_one_and_update(
        {"status": "pending"},
        {"$set": {"status": "processing", "started_at": datetime.utcnow()}},
    )
    if not task:
        return False

    try:
        refs = _load_reference(db)
        if not refs:
            raise RuntimeError("reference_tracks is empty; seed it first.")

        y, sr = _read_gridfs_audio(fs, task["gridfs_id"])
        fp = extract_features(y, sr)
        genre, conf = _predict_genre(fp, refs)

        db.results.insert_one(
            {
                "created_at": datetime.utcnow(),
                "task_id": str(task["_id"]),
                "genre_predicted": genre,
                "confidence": conf,
            }
        )
        db.tasks.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "done", "finished_at": datetime.utcnow()}}
        )
    except Exception as exc:  
        db.tasks.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "failed", "error": str(exc), "finished_at": datetime.utcnow()}}
        )
    return True

def main():
    db, fs = _connect()
    while True:
        worked = process_one(db, fs)
        if not worked:
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
