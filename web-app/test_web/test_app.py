# tests/test_app.py
# pylint: disable=import-error

from io import BytesIO


def test_index_page_with_results(client):
    """The index page should render correctly and display classifications when data exists."""
    resp = client.get("/")
    assert resp.status_code == 200

    html = resp.data.decode("utf-8")

    # Basic UI elements
    assert "Music Classification Dashboard" in html
    assert "Total Songs" in html
    assert "Rock" in html
    assert "Hiphop" in html

    # Sample data content
    assert "song_rock.mp3" in html
    assert "song_hiphop.mp3" in html

    assert "Rock" in html
    assert "Hiphop" in html


def test_index_page_no_results(monkeypatch):
    """The index page should show an empty-state message when the DB has no results."""
    import app as app_module  # pylint: disable=import-outside-toplevel

    class EmptyCursor:
        def sort(self, *_args, **_kwargs):
            return self

        def limit(self, _n):
            return []

    class EmptyCollection:
        def find(self, *_args, **_kwargs):
            return EmptyCursor()

        def count_documents(self, _filter):
            return 0

    class EmptyDB:
        def __init__(self):
            self.classifications = EmptyCollection()

    def get_empty_db():
        return EmptyDB()

    # Patch get_database() to use an empty DB
    monkeypatch.setattr(app_module, "get_database", get_empty_db)
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "No classification results yet." in html


def test_api_results(client):
    """Test JSON structure returned by /api/results."""
    resp = client.get("/api/results")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0

    first = data["results"][0]
    assert isinstance(first["_id"], str)
    assert "classification" in first
    assert "confidence" in first
    assert "timestamp" in first


def test_api_results_timestamp_format(client):
    """Test that /api/results converts datetime timestamps into ISO strings."""
    resp = client.get("/api/results")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "results" in data
    assert len(data["results"]) > 0

    first = data["results"][0]
    ts = first.get("timestamp")

    # timestamp must be a string in ISO 8601 format
    assert isinstance(ts, str)
    assert "T" in ts  # ISO timestamps contain 'T'
    assert ":" in ts


def test_api_stats(client):
    """Test /api/stats for correct counting and percentage calculation."""
    resp = client.get("/api/stats")
    assert resp.status_code == 200

    data = resp.get_json()
    total = data["total"]
    rock = data["rock"]
    hiphop = data["hiphop"]

    # Total = rock + hiphop
    assert total == rock + hiphop

    if total > 0:
        expected_rock_pct = round(rock / total * 100, 2)
        expected_hiphop_pct = round(hiphop / total * 100, 2)
        assert data["rock_percentage"] == expected_rock_pct
        assert data["hiphop_percentage"] == expected_hiphop_pct
    else:
        assert data["rock_percentage"] == 0
        assert data["hiphop_percentage"] == 0


def test_upload_audio_missing_file(client):
    """Test /api/upload-audio without file returns 400."""
    resp = client.post("/api/upload-audio")
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_upload_audio_empty_filename(client):
    """Test /api/upload-audio with empty filename returns 400."""
    data = {"audio": (BytesIO(b"test"), "")}
    resp = client.post("/api/upload-audio", data=data)
    assert resp.status_code == 400


def test_upload_audio_invalid_file(client):
    """Test /api/upload-audio with invalid file extension returns 400."""
    data = {"audio": (BytesIO(b"test data"), "wrong.txt")}
    resp = client.post("/api/upload-audio", data=data)
    assert resp.status_code == 400


def test_record_audio_missing_file(client):
    """Test /api/record-audio without file returns 400."""
    resp = client.post("/api/record-audio")
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
