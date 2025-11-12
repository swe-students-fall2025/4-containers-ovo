import numpy as np
from app.features import extract_features, cosine_sim

def test_extract_features_shape_and_norm():
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 1s A4 tone
    vec = extract_features(y, sr)
    assert vec.shape[0] in (26, 34)  
    assert 0.99 < np.linalg.norm(vec) < 1.01

def test_cosine_similarity_values():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([1.0, 0.0])
    assert abs(cosine_sim(a, b)) < 1e-6
    assert cosine_sim(a, c) > 0.9999
