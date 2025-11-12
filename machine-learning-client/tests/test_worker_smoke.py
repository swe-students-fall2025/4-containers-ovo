from unittest.mock import patch, MagicMock
import numpy as np

@patch("app.worker.extract_features", return_value=np.ones(34) / np.sqrt(34))
@patch("app.worker.GridFS")
def test_process_one_smoke(mock_fs, _mock_extract):
    import app.worker as w

    # Constructing fake databases/collections
    db = MagicMock()
    db.reference_tracks.find.return_value = [{"genre": "jazz", "fp": (np.ones(34)/np.sqrt(34)).tolist()}]
    fake_task = {"_id": "t1", "gridfs_id": "gid", "status": "pending"}
    db.tasks.find_one_and_update.return_value = fake_task

    # Have fs.get(...).read() return a short segment of pseudo-audio.
    mock_fs.return_value.get.return_value.read.return_value = b"\x00" * 100

    # Avoid actually reading the audio; directly patch the read function.
    with patch("app.worker._read_gridfs_audio", return_value=(np.ones(22050), 22050)):
        worked = w.process_one(db, mock_fs.return_value)

    assert worked is True
    db.results.insert_one.assert_called()
    db.tasks.update_one.assert_called()

