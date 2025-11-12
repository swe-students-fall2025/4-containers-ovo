"""
Script to insert sample classification data for testing.
This helps visualize the dashboard before ML client is ready.
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
import random
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Sample filenames
SAMPLE_FILENAMES = [
    "fma_small/000/000002.mp3",
    "fma_small/000/000005.mp3",
    "fma_small/001/001002.mp3",
    "fma_small/001/001005.mp3",
    "fma_small/002/002001.mp3",
    "fma_small/002/002003.mp3",
    "fma_small/003/003001.mp3",
    "fma_small/003/003004.mp3",
    "fma_small/004/004002.mp3",
    "fma_small/004/004005.mp3",
]


def insert_sample_data(num_samples=20):
    """Insert sample classification data."""
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.MONGO_DB_NAME]
    collection = db.classifications

    # Clear existing data (optional - comment out if you want to keep existing)
    # collection.delete_many({})

    # Generate sample data
    base_time = datetime.now() - timedelta(days=1)
    samples = []

    for i in range(num_samples):
        classification = random.choice(["vocal", "instrumental"])
        confidence = random.uniform(0.75, 0.99)
        filename = random.choice(SAMPLE_FILENAMES)

        sample = {
            "filename": filename,
            "filepath": f"/data/{filename}",
            "classification": classification,
            "confidence": round(confidence, 2),
            "timestamp": base_time + timedelta(hours=i),
            "features": {
                "duration": random.uniform(20, 60),
                "sample_rate": 22050,
            },
        }
        samples.append(sample)

    # Insert samples
    result = collection.insert_many(samples)
    print(f"✅ Inserted {len(result.inserted_ids)} sample records")
    print(f"   - Vocal: {sum(1 for s in samples if s['classification'] == 'vocal')}")
    print(
        f"   - Instrumental: {sum(1 for s in samples if s['classification'] == 'instrumental')}"
    )

    # Show stats
    total = collection.count_documents({})
    vocal = collection.count_documents({"classification": "vocal"})
    instrumental = collection.count_documents({"classification": "instrumental"})

    print(f"\n📊 Current database stats:")
    print(f"   - Total: {total}")
    print(f"   - Vocal: {vocal} ({vocal/total*100:.1f}%)")
    print(f"   - Instrumental: {instrumental} ({instrumental/total*100:.1f}%)")

    client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Insert sample classification data")
    parser.add_argument(
        "-n", "--num", type=int, default=20, help="Number of samples to insert"
    )
    args = parser.parse_args()

    print("🎵 Inserting sample music classification data...")
    print(f"   MongoDB URI: {Config.MONGO_URI}")
    print(f"   Database: {Config.MONGO_DB_NAME}\n")

    try:
        insert_sample_data(args.num)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure MongoDB is running:")
        print("   docker run --name mongodb -d -p 27017:27017 mongo")
        sys.exit(1)
