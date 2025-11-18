#!/usr/bin/env python3
"""
Continuously running background worker service for the ML client.

Responsibilities:
- Poll MongoDB GridFS (`fs.files`) for entries with status="pending"
- Atomically claim a task by setting status="processing"
- Load audio from GridFS, extract features, run the model, and write results
- Mark task status to "done" or "error" accordingly
- Never exit on transient failures; gracefully handle SIGTERM/SIGINT
"""

# pylint: disable=import-error,import-outside-toplevel,invalid-name

import os
import time
from datetime import datetime
import signal
import logging
from typing import Tuple, Any, Optional
from threading import Event

import numpy as np
from gridfs import GridFS
from pymongo import MongoClient
from pymongo.errors import PyMongoError, AutoReconnect, ServerSelectionTimeoutError
import joblib

try:
    # Prefer package-style import when executed as module: `python -m app.worker`
    from app.features import extract_features_audio
except ImportError:  # pragma: no cover
    # Fallback if executed directly: `python app/worker.py`
    from features import extract_features_audio


# Load trained model + scaler at module import time
MODEL_PATH = "/app/data/model_rock_hiphop.joblib"
LABEL_ENCODER_PATH = "/app/data/label_encoder.joblib"

# Model and label encoder will be loaded lazily with retry so the container
# doesn't exit if the files are not present at container start.
model = None
label_encoder = None

# Configure logging early so we can log model-load attempts
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ml-worker")


def ensure_model_loaded(initial_backoff: float = 1.0) -> None:
    """Attempt to load model and scaler, retrying with exponential backoff.

    This keeps the worker alive if the model files are not yet present
    (for example when the image is built but data files are mounted later).
    """
    global model, label_encoder
    backoff = initial_backoff
    while model is None or label_encoder is None:
        try:
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            logger.info("Loaded model from %s and label encoder from %s", MODEL_PATH, LABEL_ENCODER_PATH)
            return
        except Exception as exc:  # pragma: no cover - runtime retry behavior
            logger.warning("Model files not available yet: %s. Retrying in %.1fs", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


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


def process_one(db, gridfs_bucket: GridFS) -> bool:
    """Process a single pending classification task.

    Treats GridFS file entries (`fs.files`) with status="pending" as tasks.
    Transitions to "processing" atomically, runs inference, writes results.
    Returns True if a task was processed, False if none or on transient DB error.
    """
    # Tasks are created by the web app in the `tasks` collection and reference
    # a GridFS id in `gridfs_id`. Read from `tasks` for pending work.
    tasks_collection = db["tasks"]

    try:
        task_doc = tasks_collection.find_one_and_update(
            {"status": "pending"}, {"$set": {"status": "processing"}}
        )
    except (PyMongoError, Exception) as exc:  # pylint: disable=broad-except
        logger.warning("DB error while fetching task: %s", exc)
        return False

    if task_doc is None:
        return False

    try:
        # GridFS read audio by the stored gridfs id
        gridfs_id = task_doc.get("gridfs_id")
        audio, sample_rate = _read_gridfs_audio(gridfs_bucket, gridfs_id)

        # extract vector
        feature_vector = extract_features_audio(audio, sample_rate).reshape(1, -1)

        # Use the saved pipeline directly (it includes any scaler);
        # the saved label encoder (if present) maps numeric labels back to strings.
        try:
            proba = model.predict_proba(feature_vector)[0]
            idx = int(np.argmax(proba))
            predicted_label = model.classes_[idx]
            confidence = float(proba[idx])
        except Exception:
            predicted_label = model.predict(feature_vector)[0]
            confidence = 1.0

        # Map predicted_label to human-readable class using label_encoder if available
        try:
            predicted_genre = label_encoder.inverse_transform([predicted_label])[0]
        except Exception:
            # If inverse transform fails, fall back to string form
            predicted_genre = predicted_label

        # Update the classifications collection (web app created an initial entry)
        db.classifications.update_one(
            {"task_id": str(task_doc.get("_id"))},
            {
                "$set": {
                    "classification": predicted_genre,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow(),
                }
            },
        )

        # Mark task done
        tasks_collection.update_one(
            {"_id": task_doc["_id"]},
            {"$set": {"status": "done", "predicted_genre": predicted_genre}},
        )

        logger.info("Processed task %s file %s => %s", task_doc.get("_id"), task_doc.get("filename"), predicted_genre)
        return True

    except Exception as exc:  # pylint: disable=broad-except
        tasks_collection.update_one(
            {"_id": task_doc["_id"]},
            {"$set": {"status": "error", "error_message": str(exc)}},
        )
        logger.exception("Failed processing task %s file %s", task_doc.get("_id"), task_doc.get("filename"))
        return False


def process_loop(
    db,
    gridfs_bucket: GridFS,
    poll_interval: float,
    stop_event: Event,
) -> None:
    """Loop forever, processing tasks as they arrive, until stop_event is set."""
    idle_ticks = 0
    while not stop_event.is_set():
        worked = process_one(db, gridfs_bucket)
        if worked:
            idle_ticks = 0
            continue
        # Backoff on idle/empty queue
        idle_ticks = min(idle_ticks + 1, 10)
        time.sleep(min(poll_interval * idle_ticks, 5.0))


def create_mongo_client(uri: str) -> MongoClient:
    """Factory for MongoClient (tests may patch this)."""
    # Short server selection timeout to allow faster retry/backoff
    return MongoClient(uri, serverSelectionTimeoutMS=2000)


def main() -> None:
    """Entry point for the continuously running worker service."""
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
    mongo_db_name = os.environ.get("MONGO_DB_NAME", "ml_system")
    poll_interval = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))

    # Graceful shutdown signal handling
    stop_event = Event()

    def _handle_signal(signum, _frame):
        logger.info("Received signal %s, shutting down gracefully...", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Worker starting. MongoDB=%s DB=%s", mongo_uri, mongo_db_name)

    client: Optional[MongoClient] = None
    database = None
    gridfs_bucket: Optional[GridFS] = None

    backoff_seconds = 1.0
    while not stop_event.is_set():
        try:
            if client is None:
                # Establish or re-establish connection
                candidate = create_mongo_client(mongo_uri)
                candidate.admin.command("ping")
                client = candidate
                database = client[mongo_db_name]
                gridfs_bucket = GridFS(database)
                logger.info("Connected to MongoDB")
                # Ensure model/scaler are loaded (this will retry until available)
                ensure_model_loaded()
                backoff_seconds = 1.0

            # Run the processing loop; it returns only on stop_event
            process_loop(database, gridfs_bucket, poll_interval, stop_event)

        except (AutoReconnect, ServerSelectionTimeoutError, PyMongoError) as exc:
            logger.warning(
                "MongoDB connection issue: %s. Retrying in %.1fs", exc, backoff_seconds
            )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, 10.0)
            client = None
            continue
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Unhandled error in main loop: %s", exc)
            time.sleep(min(backoff_seconds, 5.0))
            backoff_seconds = min(backoff_seconds * 2.0, 10.0)
            continue

    logger.info("Worker stopped.")


if __name__ == "__main__":
    main()
