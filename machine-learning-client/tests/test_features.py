import numpy as np
import pytest

from app.features import (
    extract_features,
    cosine_sim,
    extract_features_audio,
)


def test_extract_features_zero_vector():
    """Zero input still produces a valid MFCC vector (normalized or zero)."""
    y = np.zeros(22050, dtype=np.float32)
    sr = 22050
    vec = extract_features(y, sr)

    assert vec.shape == (26,)
    assert np.isfinite(vec).all()

    # Either zero vector or normalized vector
    norm = np.linalg.norm(vec)
    assert norm == 0 or pytest.approx(norm, rel=1e-3) == 1.0


def test_extract_features_random_audio():
    y = np.random.randn(22050).astype(np.float32)
    sr = 22050
    vec = extract_features(y, sr)

    assert vec.shape == (26,)
    assert np.isfinite(vec).all()
    assert pytest.approx(np.linalg.norm(vec), rel=1e-3) == 1.0


def test_cosine_sim_identical():
    a = np.array([1, 2, 3], float)
    assert cosine_sim(a, a) == pytest.approx(1.0)


def test_cosine_sim_orthogonal():
    a = np.array([1, 0], float)
    b = np.array([0, 1], float)
    assert cosine_sim(a, b) == 0.0


def test_cosine_sim_zero_vector():
    a = np.array([0, 0], float)
    b = np.array([5, 1], float)
    assert cosine_sim(a, b) == 0.0


def test_extract_features_audio_mock(monkeypatch):
    """Mock librosa functions to test extract_features_audio."""
    monkeypatch.setattr("librosa.feature.rms", lambda y: np.array([[0.5]]))
    monkeypatch.setattr(
        "librosa.feature.spectral_centroid", lambda y, sr: np.array([[1000.0]])
    )
    monkeypatch.setattr(
        "librosa.feature.spectral_bandwidth", lambda y, sr: np.array([[200.0]])
    )
    monkeypatch.setattr(
        "librosa.feature.zero_crossing_rate", lambda y: np.array([[0.1]])
    )
    monkeypatch.setattr("librosa.beat.beat_track", lambda y, sr: (120.0, None))
    monkeypatch.setattr(
        "librosa.feature.spectral_flatness", lambda y: np.array([[0.3]])
    )

    y = np.random.randn(22050).astype(np.float32)
    sr = 22050
    vec = extract_features_audio(y, sr)

    assert vec.shape == (8,)
    assert np.isfinite(vec).all()
