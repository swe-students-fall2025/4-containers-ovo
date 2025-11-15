# pylint: disable=import-error
"""Quick local tests for the trained rock-vs-hiphop model."""
# test_local_model.py

import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

from app.features import extract_features_audio

# -----------------------------------------------------------
# machine-learning-client/local_test
# -----------------------------------------------------------
BASE = Path(__file__).resolve().parent  # local_test/
PROJECT_ROOT = BASE.parent  # machine-learning-client/


sys.path.append(str(PROJECT_ROOT))


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """load audio vector"""

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def main() -> None:
    """Run local tests for the trained rock vs. hip-hop model."""

    audio_path = BASE / "test_audio_hiphop.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"CAN NOT FIND AUDIO: {audio_path}")

    print(f"USE TEST AUDIO: {audio_path}")

    scaler_path = PROJECT_ROOT / "data" / "fma_metadata" / "scaler_rock_hiphop.joblib"
    model_path = PROJECT_ROOT / "data" / "fma_metadata" / "model_rock_hiphop.joblib"

    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "CAN NOT FIND scaler OR model，PLEASE CHECK PATH：\n"
            f"scaler: {scaler_path}\n"
            f"model : {model_path}"
        )

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    print("LOAD scaler AND model SUCESSFULLY")

    audio, sr = load_audio_mono(audio_path)

    feat = extract_features_audio(audio, sr)  # shape: (8,)
    feat = feat.reshape(1, -1)  # shape: (1, 8)

    print("ORIGINAL FEATURE VECTORS:", feat)

    feat_scaled = scaler.transform(feat)

    pred = model.predict(feat_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_scaled)[0]

    print("\n====== PREDICTED RESULT ======")
    print("PERDICTED CATEGORY:", pred)
    if proba is not None:
        print("PERDICT DISTRIBUTION:")
        print("classes_:", model.classes_)
        print("proba  :", proba)


if __name__ == "__main__":
    main()
