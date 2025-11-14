"""
Feature extraction helpers for ML client.
"""

# pylint: disable=invalid-name,import-error

import numpy as np
import librosa


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract normalized MFCC-based fingerprint for comparison."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26)
    vec = np.mean(mfcc, axis=1)

    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec)

    return vec / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
