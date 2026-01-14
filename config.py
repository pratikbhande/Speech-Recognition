"""Configuration settings for voice recognition POC."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
AUDIO_DIR = DATA_DIR / "audio"

# Create directories
for dir_path in [DATA_DIR, SAMPLES_DIR, AUDIO_DIR]:
    dir_path.mkdir(exist_ok=True)

# Database
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = "voice_recognition"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Voice Recognition
EMBEDDING_DIM = 192
SIMILARITY_THRESHOLD = 0.75
COLLECTION_NAME = "voice_embeddings"
MODEL_TYPE = "speechbrain"  # Options: "speechbrain" or "wespeaker"

# Audio
SAMPLE_RATE = 16000
AUDIO_FORMAT = "wav"

# Sample speakers
SAMPLE_SPEAKERS = [
    "John Smith", "Sarah Johnson", "Mike Davis", "Emily Brown",
    "David Wilson", "Lisa Anderson", "Tom Martinez", "Anna Lee",
    "Chris Taylor", "Maria Garcia"
]

SAMPLE_TEXT = "Hello, this is a voice sample for speaker recognition testing."