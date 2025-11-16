# machine-learning-client/tests/test_seed_reference.py
import numpy as np
from unittest.mock import patch, MagicMock
from app.seed_reference import seed_reference_tracks


@patch("app.seed_reference.librosa.load", return_value=(np.ones(22050), 22050))
@patch("app.seed_reference.time.time", return_value=1234567890)
def test_seed_reference_tracks(mock_time, mock_load):
    db = MagicMock()
    db.reference_tracks.insert_many.return_value = True

    seed_reference_tracks(db)

    # 确认有插入数据库
    assert db.reference_tracks.insert_many.called

    args, _ = db.reference_tracks.insert_many.call_args
    docs = args[0]

    assert len(docs) == 2  # vocal + electronic
    assert "genre" in docs[0]
    assert "fp" in docs[0]
    assert isinstance(docs[0]["fp"], list)
