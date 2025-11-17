# test_local_model.py

import sys
from pathlib import Path

import numpy as np
import joblib
import soundfile as sf

# -----------------------------------------------------------
# machine-learning-client/local_test
# -----------------------------------------------------------
BASE = Path(__file__).resolve().parent  # local_test/
PROJECT_ROOT = BASE.parent  # machine-learning-client/


sys.path.append(str(PROJECT_ROOT))


from app.features import extract_features_audio  # noqa: E402


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def main() -> None:

    audio_path = BASE / "hip.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"CAN NOT FIND AUDIO: {audio_path}")

    print(f"USE TEST AUDIO: {audio_path}")

    model_path = PROJECT_ROOT / "data" / "model_rock_hiphop.joblib"
    le_path = PROJECT_ROOT / "data" / "label_encoder.joblib"

    if not model_path.exists() or not le_path.exists():
        raise FileNotFoundError(
            "CAN NOT FIND model OR label encoder，PLEASE CHECK PATH：\n"
            f"model : {model_path}\n"
            f"le    : {le_path}"
        )

    # Load pipeline (contains StandardScaler + RandomForestClassifier)
    pipeline = joblib.load(model_path)
    le = joblib.load(le_path)

    print("\n====== MODEL INFO ======")
    if hasattr(pipeline, 'named_steps'):
        print("PIPELINE STEPS:", list(pipeline.named_steps.keys()))
    print("LABEL ENCODER CLASSES:", le.classes_)
    print("LOAD model SUCCESSFULLY")

    print("\n====== LOAD AUDIO ======")
    audio, sr = load_audio_mono(audio_path)
    print(f"Audio shape: {audio.shape}, Sample rate: {sr}")

    print("\n====== EXTRACT FEATURES ======")
    feat = extract_features_audio(audio, sr)  # shape: (8,)
    print("FEATURE VECTORS:", feat)
    feat_batch = feat.reshape(1, -1)  # shape: (1, 8)
    print("BATCH SHAPE:", feat_batch.shape)

    print("\n====== PREDICT ======")
    # Use pipeline to transform and predict (scaler is inside pipeline)
    pred_encoded = pipeline.predict(feat_batch)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    print(f"PREDICTED CLASS (encoded): {pred_encoded}")
    print(f"PREDICTED CLASS (label): {pred_label}")

    # Get probabilities from the RandomForest step
    rf_model = pipeline.named_steps.get('randomforestclassifier')
    if rf_model is not None:
        scaler = pipeline.named_steps.get('standardscaler')
        feat_scaled = scaler.transform(feat_batch)
        proba = rf_model.predict_proba(feat_scaled)[0]
        print("\nPREDICT PROBABILITIES:")
        for i, class_name in enumerate(le.classes_):
            print(f"  {class_name}: {proba[i]:.4f}")

    print("\n====== TEST COMPLETE ======")


if __name__ == "__main__":
    main()
