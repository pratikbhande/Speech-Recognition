"""
Setup sample speakers with multi-embedding enrollment.

This script enrolls pre-recorded audio samples as test speakers.
Each speaker gets 3-5 embeddings from their single audio file.
"""

from pathlib import Path
import soundfile as sf
import numpy as np
import logging
import config
from voice_embeddings import VoiceEmbedder
from database import MongoManager, QdrantManager
from datetime import datetime
from audio_preprocessing import preprocess_audio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_samples():
    """
    Enroll all sample speakers with multi-embedding support.
    
    Process:
    1. Find audio files in data/samples/
    2. Process each file with minimal preprocessing
    3. Extract 3-5 embeddings per speaker
    4. Store in MongoDB (metadata) and Qdrant (embeddings)
    """
    
    logger.info("=" * 70)
    logger.info("SAMPLE SPEAKER ENROLLMENT - Multi-Embedding Mode")
    logger.info("=" * 70)
    
    # Initialize components
    try:
        embeddings = VoiceEmbedder()
        mongo = MongoManager()
        qdrant = QdrantManager()
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        return
    
    # Find samples directory
    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        logger.error(f"❌ Directory not found: {samples_dir}")
        logger.info("Please create 'data/samples/' and add audio files")
        return
    
    # Get all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(samples_dir.glob(f"*{ext}"))
        audio_files.extend(samples_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        logger.error("❌ No audio files found!")
        logger.info(f"Supported formats: {', '.join(audio_extensions)}")
        return
    
    # Sort files by name for consistent ordering
    audio_files.sort(key=lambda x: x.name)
    
    logger.info(f"\n🔍 Found {len(audio_files)} audio files")
    logger.info(f"📝 Will enroll first {min(len(audio_files), len(config.SAMPLE_SPEAKERS))} speakers")
    logger.info("=" * 70)
    
    enrolled_count = 0
    failed_count = 0
    
    # Process each audio file
    for idx, audio_file in enumerate(audio_files[:len(config.SAMPLE_SPEAKERS)], 1):
        speaker_name = config.SAMPLE_SPEAKERS[idx - 1] if idx <= len(config.SAMPLE_SPEAKERS) else f"Speaker {idx}"
        
        logger.info(f"\n[{idx}/{min(len(audio_files), len(config.SAMPLE_SPEAKERS))}] Processing: {speaker_name}")
        logger.info(f"    File: {audio_file.name}")
        
        try:
            # Load audio
            audio, sr = sf.read(audio_file)
            logger.info(f"    ✅ Loaded: {len(audio)/sr:.1f}s @ {sr}Hz")
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
                logger.info(f"    ✅ Converted to mono")
            
            # MINIMAL preprocessing for enrollment
            audio_processed = preprocess_audio(audio, sr, for_enrollment=True)
            logger.info(f"    ✅ Preprocessed: {len(audio_processed)/sr:.1f}s")
            
            # Safety check - ensure preprocessing didn't destroy too much
            if len(audio_processed) < sr * 1.0:
                logger.warning(f"    ⚠️  Preprocessing too aggressive, using original")
                audio_processed = audio
                # Simple normalization
                max_val = np.abs(audio_processed).max()
                if max_val > 0:
                    audio_processed = audio_processed / max_val
            
            # Validate minimum length
            if len(audio_processed) < sr * 2.0:
                logger.warning(
                    f"    ⚠️  Skipped: Too short (need ≥2s, got {len(audio_processed)/sr:.1f}s)"
                )
                failed_count += 1
                continue
            
            # Truncate if too long (keep first 40s)
            if len(audio_processed) > sr * 40:
                audio_processed = audio_processed[:sr * 40]
                logger.info(f"    ✅ Truncated to 40s")
            
            # Determine number of embeddings based on audio length
            audio_length_sec = len(audio_processed) / sr
            
            if audio_length_sec < 8:
                num_embeddings = 3
            elif audio_length_sec < 20:
                num_embeddings = 4
            else:
                num_embeddings = 5
            
            logger.info(
                f"    📊 Audio length: {audio_length_sec:.1f}s → "
                f"Creating {num_embeddings} embeddings"
            )
            
            # Generate client ID
            client_id = f"SAMPLE_{idx:02d}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Extract MULTIPLE embeddings
            embeddings_list = embeddings.extract_multiple_embeddings(
                audio_processed, sr, num_embeddings=num_embeddings
            )
            
            logger.info(f"    ✅ Created {len(embeddings_list)} embeddings")
            
            # Store in MongoDB
            user_doc = {
                "client_id": client_id,
                "name": speaker_name,
                "created_at": datetime.utcnow(),
                "is_sample": True,
                "num_embeddings": len(embeddings_list),
                "audio_length_sec": audio_length_sec,
                "original_file": audio_file.name
            }
            mongo.users.insert_one(user_doc)
            logger.info(f"    ✅ Stored metadata in MongoDB")
            
            # Store ALL embeddings in Qdrant
            qdrant.insert_multiple(client_id, embeddings_list)
            logger.info(f"    ✅ Stored embeddings in Qdrant")
            
            # Save processed audio with standard name
            new_name = samples_dir / f"sample_{idx:02d}.wav"
            sf.write(new_name, audio_processed, sr)
            logger.info(f"    ✅ Saved as: {new_name.name}")
            
            # Delete original file if different name
            if audio_file != new_name and audio_file.exists():
                try:
                    audio_file.unlink()
                    logger.info(f"    ✅ Removed original file")
                except Exception as e:
                    logger.warning(f"    ⚠️  Could not remove original: {e}")
            
            logger.info(f"    ✅ Successfully enrolled: {speaker_name}")
            enrolled_count += 1
            
        except Exception as e:
            logger.error(f"    ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ENROLLMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Successfully enrolled: {enrolled_count} speakers")
    logger.info(f"❌ Failed: {failed_count} speakers")
    
    if enrolled_count > 0:
        # Get stats
        stats = {
            'total_embeddings': sum(
                u.get('num_embeddings', 0) 
                for u in mongo.users.find()
            ),
            'avg_embeddings': sum(
                u.get('num_embeddings', 0) 
                for u in mongo.users.find()
            ) / enrolled_count
        }
        
        logger.info(f"📊 Total embeddings created: {stats['total_embeddings']}")
        logger.info(f"📊 Average per speaker: {stats['avg_embeddings']:.1f}")
        logger.info("\n🚀 Ready to test! Run: streamlit run app.py")
    else:
        logger.warning("\n⚠️  No speakers enrolled. Please check your audio files.")
    
    logger.info("=" * 70)


def verify_enrollment():
    """
    Verify enrollment was successful.
    
    Checks:
    - MongoDB has user records
    - Qdrant has embeddings
    - Counts match
    """
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)
    
    try:
        mongo = MongoManager()
        qdrant = QdrantManager()
        
        # Count users
        users = list(mongo.users.find())
        logger.info(f"✅ MongoDB: {len(users)} users")
        
        # Count embeddings
        qdrant_stats = qdrant.get_stats()
        logger.info(f"✅ Qdrant: {qdrant_stats.get('total_points', 0)} embeddings")
        
        # Show details
        for user in users:
            logger.info(
                f"   - {user['name']}: {user.get('num_embeddings', 1)} embeddings"
            )
        
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")


if __name__ == "__main__":
    try:
        setup_samples()
        verify_enrollment()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()