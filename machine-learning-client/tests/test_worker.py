"""
Tests for worker task processing.
"""

# pylint: disable=duplicate-code

from unittest.mock import patch, MagicMock
import numpy as np
from app.worker import process_one


@patch("app.worker.extract_features", return_value=np.ones(26))
@patch("app.worker._read_gridfs_audio", return_value=(np.ones(22050), 22050))
def test_process_one_success(_mock_read, _mock_extract):
    """
    Ensure worker processes a pending task and stores results.
    """
    mock_db = MagicMock()
    mock_db.tasks.find_one_and_update.return_value = {
        "_id": "task1",
        "gridfs_id": "audio123",
        "status": "pending",
    }

    mock_db.reference_tracks.find.return_value = [
        {"genre": "rock", "fp": np.ones(26).tolist()}
    ]

    mock_db.results.insert_one.return_value = True
    mock_fs = MagicMock()

    did_work = process_one(mock_db, mock_fs)

    assert did_work is True
    mock_db.results.insert_one.assert_called_once()
    mock_db.tasks.update_one.assert_called()


def test_process_one_no_task():
    """
    Ensure worker returns False when no tasks are available.
    """
    mock_db = MagicMock()
    mock_db.tasks.find_one_and_update.return_value = None
    mock_fs = MagicMock()

    did_work = process_one(mock_db, mock_fs)
    assert did_work is False
