"""
Flask web application for displaying music classification results.
Music classification: Vocal vs Instrumental.
"""

from datetime import datetime

from flask import Flask, render_template, jsonify

from database import get_database

app = Flask(__name__)
app.config.from_object("config.Config")


@app.route("/")
def index():
    """Display the main dashboard page with music classification results."""
    db = get_database()
    # Get recent classification results from ML client
    # Expected data structure:
    # {
    #   "filename": "song.mp3",
    #   "filepath": "/path/to/song.mp3",
    #   "classification": "vocal" or "instrumental",
    #   "confidence": 0.95,
    #   "timestamp": datetime,
    #   "features": {...}  # optional ML features
    # }
    recent_results = list(db.classifications.find().sort("timestamp", -1).limit(20))

    # Get statistics
    total_count = db.classifications.count_documents({})
    vocal_count = db.classifications.count_documents({"classification": "vocal"})
    instrumental_count = db.classifications.count_documents(
        {"classification": "instrumental"}
    )

    stats = {
        "total": total_count,
        "vocal": vocal_count,
        "instrumental": instrumental_count,
    }

    return render_template("index.html", results=recent_results, stats=stats)


@app.route("/api/results")
def api_results():
    """API endpoint to get classification results."""
    db = get_database()
    limit = int(app.config.get("API_LIMIT", 100))
    results = list(db.classifications.find().sort("timestamp", -1).limit(limit))

    # Convert ObjectId and datetime to string for JSON serialization
    for item in results:
        item["_id"] = str(item["_id"])
        if "timestamp" in item and isinstance(item["timestamp"], datetime):
            item["timestamp"] = item["timestamp"].isoformat()

    return jsonify({"results": results})


@app.route("/api/stats")
def api_stats():
    """API endpoint to get classification statistics."""
    db = get_database()
    total = db.classifications.count_documents({})
    vocal = db.classifications.count_documents({"classification": "vocal"})
    instrumental = db.classifications.count_documents(
        {"classification": "instrumental"}
    )

    return jsonify(
        {
            "total": total,
            "vocal": vocal,
            "instrumental": instrumental,
            "vocal_percentage": round(vocal / total * 100, 2) if total > 0 else 0,
            "instrumental_percentage": (
                round(instrumental / total * 100, 2) if total > 0 else 0
            ),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
