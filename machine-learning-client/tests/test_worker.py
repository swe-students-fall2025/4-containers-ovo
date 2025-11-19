import numpy as np
import pytest
from unittest.mock import MagicMock
from pymongo.errors import PyMongoError

from app.worker import (
    process_one,
    process_loop,
    ensure_model_loaded,
    create_mongo_client,
    _read_gridfs_audio,
    main,
)
import builtins


# -----------------------------------------------------------------------------
# Fake audio reader
# -----------------------------------------------------------------------------
def fake_audio_reader(gridfs_bucket, gridfs_id):
    return np.ones(22050, dtype=np.float32), 22050


# -----------------------------------------------------------------------------
# Fake DB supporting both dict and attribute access
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_db():
    class FakeCollection:
        def __init__(self):
            self.pending = {
                "_id": "123",
                "status": "pending",
                "gridfs_id": "abc",
                "filename": "song.wav",
            }
            self.updated = None

        def find_one_and_update(self, query, update):
            if self.pending and self.pending["status"] == "pending":
                self.pending["status"] = "processing"
                return self.pending
            return None

        def update_one(self, query, update):
            self.updated = update

    class FakeDB:
        def __init__(self):
            self._storage = {
                "tasks": FakeCollection(),
                "classifications": MagicMock(),
            }

        def __getitem__(self, key):
            return self._storage[key]

        def __getattr__(self, key):
            if key in self._storage:
                return self._storage[key]
            raise AttributeError(key)

    return FakeDB()


@pytest.fixture
def mock_gridfs():
    return MagicMock()


# -----------------------------------------------------------------------------
# process_one - success case
# -----------------------------------------------------------------------------
def test_process_one_success(mock_db, mock_gridfs, monkeypatch):
    monkeypatch.setattr("app.worker._read_gridfs_audio", fake_audio_reader)

    class FakeModel:
        classes_ = np.array(["rock", "hiphop"])

        def predict_proba(self, vec):
            return np.array([[0.2, 0.8]])

    class FakeEncoder:
        def inverse_transform(self, labels):
            return labels

    monkeypatch.setattr("app.worker.model", FakeModel())
    monkeypatch.setattr("app.worker.label_encoder", FakeEncoder())

    worked = process_one(mock_db, mock_gridfs)
    assert worked is True
    assert mock_db.tasks.updated["$set"]["status"] == "done"


# -----------------------------------------------------------------------------
# process_one - no task
# -----------------------------------------------------------------------------
def test_process_one_no_task(mock_db, mock_gridfs):
    mock_db.tasks.pending = None
    assert process_one(mock_db, mock_gridfs) is False


# -----------------------------------------------------------------------------
# process_one - DB error
# -----------------------------------------------------------------------------
def test_process_one_db_error(mock_db, mock_gridfs, monkeypatch):
    def bad_find(*args, **kwargs):
        raise PyMongoError("DB err")

    monkeypatch.setattr(mock_db.tasks, "find_one_and_update", bad_find)
    assert process_one(mock_db, mock_gridfs) is False


# -----------------------------------------------------------------------------
# process_one - label_encoder inverse_transform fails
# -----------------------------------------------------------------------------
def test_process_one_label_encoder_fallback(mock_db, mock_gridfs, monkeypatch):
    monkeypatch.setattr("app.worker._read_gridfs_audio", fake_audio_reader)

    class FakeModel:
        classes_ = np.array(["rock"])

        def predict_proba(self, vec):
            return np.array([[1.0]])

    class BadEncoder:
        def inverse_transform(self, labels):
            raise ValueError("bad")

    monkeypatch.setattr("app.worker.model", FakeModel())
    monkeypatch.setattr("app.worker.label_encoder", BadEncoder())

    worked = process_one(mock_db, mock_gridfs)
    assert worked is True
    # fallback path writes predicted_label directly
    assert mock_db.tasks.updated["$set"]["status"] == "done"


# -----------------------------------------------------------------------------
# ensure_model_loaded - FileNotFoundError path (retry once then success)
# -----------------------------------------------------------------------------
def test_ensure_model_loaded_with_retry(monkeypatch):
    calls = {"i": 0}

    def fake_load(path):
        if calls["i"] == 0:
            calls["i"] += 1
            raise FileNotFoundError("missing")
        else:
            return "OK"

    monkeypatch.setattr("joblib.load", fake_load)

    m, le = ensure_model_loaded(initial_backoff=0.001)
    assert m == "OK"
    assert le == "OK"


# -----------------------------------------------------------------------------
# process_loop - backoff path (two iterations)
# -----------------------------------------------------------------------------
def test_process_loop_backoff(monkeypatch):
    sequence = [False, False, True]  # exit on 3rd call

    def fake_process_one(db, bucket):
        return sequence.pop(0)

    monkeypatch.setattr("app.worker.process_one", fake_process_one)

    class StopEvent:
        def is_set(self):
            return len(sequence) == 0

    process_loop(MagicMock(), MagicMock(), poll_interval=0.001, stop_event=StopEvent())
    assert True


# -----------------------------------------------------------------------------
# _read_gridfs_audio coverage
# -----------------------------------------------------------------------------
def test_read_gridfs_audio(monkeypatch):
    class FakeFSFile:
        def read(self):
            return b"FAKE"

    class FakeBucket:
        def get(self, file_id):
            return FakeFSFile()

    class FakeSoundFile:
        def __init__(self, buf):
            self.samplerate = 16000

        def read(self, dtype):
            return np.ones(100, dtype=dtype)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("soundfile.SoundFile", FakeSoundFile)

    a, sr = _read_gridfs_audio(FakeBucket(), "id")
    assert a.shape == (100,)
    assert sr == 16000


# -----------------------------------------------------------------------------
# create_mongo_client
# -----------------------------------------------------------------------------
def test_create_mongo_client(monkeypatch):
    class FakeClient:
        def __init__(self, uri, serverSelectionTimeoutMS):
            self.uri = uri

    monkeypatch.setattr("app.worker.MongoClient", FakeClient)
    c = create_mongo_client("mongodb://fake")
    assert isinstance(c, FakeClient)


# -----------------------------------------------------------------------------
# main() - simulate immediate stop
# -----------------------------------------------------------------------------
def test_main_single_iteration(monkeypatch):

    # patch create mongo client
    class FakeClient:
        def __init__(self, uri, serverSelectionTimeoutMS):
            self.admin = MagicMock()
            self.admin.command = lambda x: True

        def __getitem__(self, name):
            return MagicMock()

    monkeypatch.setattr("app.worker.MongoClient", FakeClient)
    monkeypatch.setenv("MONGO_URI", "mongodb://fake")
    monkeypatch.setenv("MONGO_DB_NAME", "db")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "0")

    # stop event will trigger after one loop
    class FakeEvent:
        called = False

        def is_set(self):
            if not self.called:
                self.called = True
                return False
            return True

        def set(self):
            pass

    monkeypatch.setattr("app.worker.Event", lambda: FakeEvent())
    monkeypatch.setattr("app.worker.ensure_model_loaded", lambda: ("M", "E"))
    monkeypatch.setattr("app.worker.process_loop", lambda *args: None)

    # run main one iteration
    main()
    assert True
