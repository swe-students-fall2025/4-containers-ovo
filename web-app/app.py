"""
Flask web application for displaying music classification results.
Music classification: Rock vs Hiphop.
"""

from datetime import datetime

from flask import Flask, render_template, jsonify, request
from gridfs import GridFS
from werkzeug.utils import secure_filename

from config import Config
from database import get_database, get_client

app = Flask(__name__)
app.config.from_object("config.Config")

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac"}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Display the main dashboard page with music classification results."""
    try:
        db = get_database()
        # Get recent classification results from ML client
        # Expected data structure:
        # {
        #   "filename": "song.mp3",
        #   "filepath": "/path/to/song.mp3",
        #   "classification": "rock" or "hiphop",
        #   "confidence": 0.95,
        #   "timestamp": datetime,
        #   "features": {...}  # optional ML features
        # }
        recent_results = list(db.classifications.find().sort("timestamp", -1).limit(20))

        # Get statistics
        total_count = db.classifications.count_documents({})
        rock_count = db.classifications.count_documents({"classification": "rock"})
        hiphop_count = db.classifications.count_documents({"classification": "hiphop"})

        stats = {
            "total": total_count,
            "rock": rock_count,
            "hiphop": hiphop_count,
        }
    except Exception as e:
        # If MongoDB is not available, show empty state
        print(f"Database connection error: {e}")
        recent_results = []
        stats = {
            "total": 0,
            "rock": 0,
            "hiphop": 0,
        }

    return render_template("index.html", results=recent_results, stats=stats)


@app.route("/api/results")
def api_results():
    """API endpoint to get classification results."""
    try:
        db = get_database()
        limit = int(app.config.get("API_LIMIT", 100))
        results = list(db.classifications.find().sort("timestamp", -1).limit(limit))

        # Convert ObjectId and datetime to string for JSON serialization
        for item in results:
            item["_id"] = str(item["_id"])
            if "timestamp" in item and isinstance(item["timestamp"], datetime):
                item["timestamp"] = item["timestamp"].isoformat()
    except Exception as e:
        print(f"Database connection error: {e}")
        results = []

    return jsonify({"results": results})


@app.route("/api/stats")
def api_stats():
    """API endpoint to get classification statistics."""
    try:
        db = get_database()
        total = db.classifications.count_documents({})
        rock = db.classifications.count_documents({"classification": "rock"})
        hiphop = db.classifications.count_documents({"classification": "hiphop"})
    except Exception as e:
        print(f"Database connection error: {e}")
        total = rock = hiphop = 0

    return jsonify(
        {
            "total": total,
            "rock": rock,
            "hiphop": hiphop,
            "rock_percentage": round(rock / total * 100, 2) if total > 0 else 0,
            "hiphop_percentage": (
                round(hiphop / total * 100, 2) if total > 0 else 0
            ),
        }
    )


@app.route("/api/record-audio", methods=["POST"])
def record_audio():
    """API endpoint to receive recorded audio and create classification task."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No audio file selected"}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        db = get_database()
        client = get_client()
        database = client[Config.MONGO_DB_NAME]

        # Store audio in GridFS
        gridfs_bucket = GridFS(database)
        audio_data = audio_file.read()
        gridfs_id = gridfs_bucket.put(
            audio_data, filename=secure_filename(audio_file.filename)
        )

        # Create classification task
        task = {
            "filename": secure_filename(audio_file.filename),
            "gridfs_id": gridfs_id,
            "status": "pending",
            "source": "audio_recording",
            "timestamp": datetime.utcnow(),
        }
        task_id = db.tasks.insert_one(task).inserted_id

        # Also create a classification entry for display (will be updated by ML client)
        classification = {
            "filename": secure_filename(audio_file.filename),
            "classification": "pending",
            "confidence": 0.0,
            "timestamp": datetime.utcnow(),
            "task_id": str(task_id),
            "source": "audio_recording",
        }
        db.classifications.insert_one(classification)

        return jsonify(
            {
                "success": True,
                "task_id": str(task_id),
                "message": "Audio uploaded and classification task created",
            }
        ), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-audio", methods=["POST"])
def upload_audio():
    """API endpoint to upload audio file and create classification task."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No audio file selected"}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        db = get_database()
        client = get_client()
        database = client[Config.MONGO_DB_NAME]

        # Store audio in GridFS
        gridfs_bucket = GridFS(database)
        audio_data = audio_file.read()
        gridfs_id = gridfs_bucket.put(
            audio_data, filename=secure_filename(audio_file.filename)
        )

        # Create classification task
        task = {
            "filename": secure_filename(audio_file.filename),
            "gridfs_id": gridfs_id,
            "status": "pending",
            "source": "file_upload",
            "timestamp": datetime.utcnow(),
        }
        task_id = db.tasks.insert_one(task).inserted_id

        # Also create a classification entry for display (will be updated by ML client)
        classification = {
            "filename": secure_filename(audio_file.filename),
            "classification": "pending",
            "confidence": 0.0,
            "timestamp": datetime.utcnow(),
            "task_id": str(task_id),
            "source": "file_upload",
        }
        db.classifications.insert_one(classification)

        return jsonify(
            {
                "success": True,
                "task_id": str(task_id),
                "message": "Audio uploaded and classification task created",
            }
        ), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
