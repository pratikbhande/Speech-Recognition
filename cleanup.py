"""Remove old sample data."""

import shutil
from pathlib import Path
import config
from database import MongoManager, QdrantManager


def cleanup_old_data():
    """Remove all old sample data."""
    
    print("🧹 Cleaning up old data...")
    
    # 1. Delete old audio files
    if config.SAMPLES_DIR.exists():
        shutil.rmtree(config.SAMPLES_DIR)
        config.SAMPLES_DIR.mkdir(exist_ok=True)
        print("✅ Deleted old audio files")
    
    # 2. Clear MongoDB
    mongo = MongoManager()
    mongo.users.delete_many({})
    print("✅ Cleared MongoDB users")
    
    # 3. Reset Qdrant collection
    qdrant = QdrantManager()
    qdrant.delete_all()
    print("✅ Reset Qdrant embeddings")
    
    print("\n🎉 Cleanup complete! Ready for fresh data.")


if __name__ == "__main__":
    cleanup_old_data()