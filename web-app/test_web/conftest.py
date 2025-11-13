# tests/conftest.py

import sys
from pathlib import Path
import pytest
from datetime import datetime

# Ensure the parent directory (where app.py lives) is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as app_module


class FakeCursor:
    """A simple cursor object that mimics PyMongo's chained sort().limit()."""
    def __init__(self, docs):
        self._docs = docs

    def sort(self, *args, **kwargs):
        return self

    def limit(self, n):
        return self._docs[:n]


class FakeCollection:
    """Fake MongoDB collection for testing without a real database."""
    def __init__(self, docs):
        self._docs = docs

    def find(self, *args, **kwargs):
        return FakeCursor(self._docs)

    def count_documents(self, filter_):
        # If filter is {}, return total number of docs
        if not filter_:
            return len(self._docs)
        cls = filter_.get("classification")
        return sum(1 for d in self._docs if d.get("classification") == cls)


class FakeDB:
    """Fake database object holding a single fake collection."""
    def __init__(self, docs):
        self.classifications = FakeCollection(docs)


@pytest.fixture
def sample_docs():
    """Basic sample classification results used across multiple tests."""
    return [
        {
            "_id": "1",
            "filename": "song_vocal.mp3",
            "classification": "vocal",
            "confidence": 0.9,
            "timestamp": datetime(2025, 1, 1, 12, 0, 0),
        },
        {
            "_id": "2",
            "filename": "song_instr.mp3",
            "classification": "instrumental",
            "confidence": 0.8,
            "timestamp": datetime(2025, 1, 1, 13, 0, 0),
        },
    ]


@pytest.fixture
def fake_db(sample_docs):
    """Return FakeDB instance populated with sample documents."""
    return FakeDB(sample_docs)


@pytest.fixture
def app(fake_db, monkeypatch):
    """Patch get_database() so the Flask app uses our fake DB instead of Mongo."""
    monkeypatch.setattr(app_module, "get_database", lambda: fake_db)

    app_module.app.config["TESTING"] = True
    return app_module.app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()
