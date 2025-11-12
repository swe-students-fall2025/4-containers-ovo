import numpy as np
import librosa

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype("float32", copy=False)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)

    vec = np.concatenate([mfcc.mean(axis=1), chroma.mean(axis=1), sc.mean(axis=1)])
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("cosine_sim expects 1D vectors")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
