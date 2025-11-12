"""
Unit tests for Flask web application.
"""

import pytest
from app import app
from database import get_database
from datetime import datetime


@pytest.fixture
def client():
    """Create a test client."""
    app.config["TESTING"] = True
    app.config["MONGO_URI"] = "mongodb://localhost:27017/test_db"
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def sample_data():
    """Create sample classification data for testing."""
    db = get_database()
    sample = [
        {
            "filename": "song1.mp3",
            "filepath": "/data/song1.mp3",
            "classification": "vocal",
            "confidence": 0.95,
            "timestamp": datetime.now(),
        },
        {
            "filename": "song2.mp3",
            "filepath": "/data/song2.mp3",
            "classification": "instrumental",
            "confidence": 0.87,
            "timestamp": datetime.now(),
        },
    ]
    db.classifications.insert_many(sample)
    yield sample
    db.classifications.delete_many({})


def test_index_route(client):
    """Test the index route."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Music Classification Dashboard" in response.data


def test_api_results_route(client):
    """Test the API results route."""
    response = client.get("/api/results")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "results" in json_data
    assert isinstance(json_data["results"], list)


def test_api_stats_route(client, sample_data):
    """Test the API stats route."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "total" in json_data
    assert "vocal" in json_data
    assert "instrumental" in json_data
    assert "vocal_percentage" in json_data
    assert "instrumental_percentage" in json_data
    assert json_data["total"] == 2
    assert json_data["vocal"] == 1
    assert json_data["instrumental"] == 1
