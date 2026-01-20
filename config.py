"""
Configuration settings for voice recognition POC.

This file contains all configurable parameters for the system.
Adjust these values based on your deployment environment and requirements.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
AUDIO_DIR = DATA_DIR / "audio"

# Create directories on import
for dir_path in [DATA_DIR, SAMPLES_DIR, AUDIO_DIR]:
    dir_path.mkdir(exist_ok=True)


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# MongoDB settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = "voice_recognition"

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))


# ============================================================================
# VOICE RECOGNITION SETTINGS
# ============================================================================

# Embedding configuration
EMBEDDING_DIM = 192  # ECAPA-TDNN embedding dimension
COLLECTION_NAME = "voice_embeddings"

# Similarity threshold for speaker identification
# UPDATED: Lowered from 0.90 to 0.75 for better recall with multi-embeddings
# 
# Threshold guide:
# - 0.85+: Very confident match (same person, same session)
# - 0.75-0.85: Confident match (same person, different conditions)
# - 0.65-0.75: Possible match (requires confirmation)
# - <0.65: Likely different person
SIMILARITY_THRESHOLD = 0.75

# Multi-embedding configuration
# Number of embeddings to create per speaker during enrollment
# Dynamically adjusted based on audio length:
# - Short audio (<8s): 3 embeddings
# - Medium audio (8-20s): 4 embeddings  
# - Long audio (>20s): 5 embeddings
MIN_EMBEDDINGS_PER_SPEAKER = 3
MAX_EMBEDDINGS_PER_SPEAKER = 5


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

# Audio configuration
SAMPLE_RATE = 16000  # Target sample rate (Hz)
AUDIO_FORMAT = "wav"  # Output format for saved audio

# Minimum audio lengths
MIN_ENROLLMENT_LENGTH_SEC = 2.0  # Minimum for enrollment
MIN_IDENTIFICATION_LENGTH_SEC = 1.0  # Minimum for identification

# Maximum audio length (for truncation)
MAX_AUDIO_LENGTH_SEC = 40.0


# ============================================================================
# SAMPLE DATA
# ============================================================================

# Sample speakers for testing/demo
SAMPLE_SPEAKERS = [
    "John Smith",
    "Sarah Johnson", 
    "Mike Davis",
    "Emily Brown",
    "David Wilson",
    "Lisa Anderson",
    "Tom Martinez",
    "Anna Lee",
    "Chris Taylor",
    "Maria Garcia"
]

# Sample text for TTS generation (if needed)
SAMPLE_TEXT = "Hello, this is a voice sample for speaker recognition testing."


# ============================================================================
# LOGGING
# ============================================================================

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Vector search settings
VECTOR_SEARCH_LIMIT = 50  # Max vectors to retrieve per search
TOP_MATCHES_LIMIT = 5  # Max users to return in results

# Batch processing
BATCH_SIZE = 10  # Batch size for bulk operations


# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features
ENABLE_NOISE_REDUCTION = True  # Apply noise reduction in preprocessing
ENABLE_AUGMENTATION = True  # Use audio augmentation for embeddings
ENABLE_QUALITY_CHECKS = True  # Validate audio quality during enrollment


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    
    errors = []
    
    # Check threshold is valid
    if not 0.0 <= SIMILARITY_THRESHOLD <= 1.0:
        errors.append(f"SIMILARITY_THRESHOLD must be between 0 and 1, got {SIMILARITY_THRESHOLD}")
    
    # Check embedding dimension
    if EMBEDDING_DIM != 192:
        errors.append(f"EMBEDDING_DIM must be 192 for ECAPA-TDNN, got {EMBEDDING_DIM}")
    
    # Check minimum lengths
    if MIN_ENROLLMENT_LENGTH_SEC < 1.0:
        errors.append(f"MIN_ENROLLMENT_LENGTH_SEC too short: {MIN_ENROLLMENT_LENGTH_SEC}s")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


# Validate on import
validate_config()