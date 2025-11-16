# machine-learning-client/tests/test_features.py
import numpy as np
from app.features import extract_features, cosine_sim


def test_extract_features_normalized():
    y = np.ones(22050)
    sr = 22050

    vec = extract_features(y, sr)

    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == 26
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_extract_features_zero_norm():
    y = np.zeros(22050)
    sr = 22050

    vec = extract_features(y, sr)
    assert np.all(vec == 0)


def test_cosine_sim():
    a = np.array([1, 0, 0], dtype=float)
    b = np.array([1, 0, 0], dtype=float)
    c = np.array([0, 1, 0], dtype=float)

    assert cosine_sim(a, b) == 1.0
    assert np.isclose(cosine_sim(a, c), 0.0)
