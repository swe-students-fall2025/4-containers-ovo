"""
Feature extraction utilities.
"""
# pylint: disable=import-error
import numpy as np
import librosa


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a stable feature vector (length 26 or 34) from a waveform.

    Returns a normalized feature vector with L2 norm ≈ 1.
    """
    # 13 MFCCs + 13 deltas = 26 features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)

    delta = librosa.feature.delta(mfcc)
    delta_mean = delta.mean(axis=1)

    vec = np.concatenate([mfcc_mean, delta_mean])

    # Normalize so vec has length ≈ 1
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between vectors.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
