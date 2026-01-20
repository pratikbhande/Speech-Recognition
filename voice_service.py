"""
Voice recognition service with multi-embedding support.

Main service for:
- Speaker enrollment with multiple embeddings
- Speaker identification using max-score matching
- User management
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import logging
import config
from voice_embeddings import VoiceEmbedder
from database import MongoManager, QdrantManager
from datetime import datetime
from audio_preprocessing import preprocess_audio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceService:
    """
    Main service for voice recognition operations.
    
    Features:
    - Multi-embedding enrollment for robustness
    - Max-score identification for better accuracy
    - Graceful fallbacks and error handling
    """
    
    def __init__(self):
        """Initialize voice service with required components."""
        try:
            self.embeddings = VoiceEmbedder()
            self.mongo = MongoManager()
            self.qdrant = QdrantManager()
            logger.info("✅ VoiceService initialized successfully")
        except Exception as e:
            logger.error(f"❌ VoiceService initialization failed: {e}")
            raise
        
    def identify_speaker(self, audio_path: str):
        """
        Identify speaker from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_known, client_id, name, confidence)
        """
        try:
            audio, sr = sf.read(audio_path)
            return self.identify_from_array(audio, sr)
        except Exception as e:
            logger.error(f"❌ Error identifying from file: {e}")
            return False, None, "Error", 0.0
    
    def identify_from_array(self, audio: np.ndarray, sr: int = 16000):
        """
        Identify speaker from audio array.
        
        Uses max-score strategy across multiple stored embeddings for
        improved robustness to voice variation, noise, and recording conditions.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate (default: 16000)
            
        Returns:
            Tuple of (is_known, client_id, name, confidence_score)
            
        Examples:
            >>> is_known, client_id, name, score = service.identify_from_array(audio)
            >>> if is_known:
            >>>     print(f"Identified: {name} with {score:.2%} confidence")
        """
        try:
            # LIGHT preprocessing for identification
            # We use minimal processing to preserve voice characteristics
            audio_processed = preprocess_audio(audio, sr, for_enrollment=False)
            
            # Validate audio length
            min_length = sr * 1.0  # 1 second minimum
            if len(audio_processed) < min_length:
                logger.warning(
                    f"Audio too short after preprocessing: {len(audio_processed)/sr:.2f}s"
                )
                # Fallback to original if preprocessing removed too much
                audio_processed = audio
                
                # Still too short? Return unknown
                if len(audio_processed) < min_length:
                    logger.warning("Audio too short even before preprocessing")
                    return False, None, "Unknown Speaker (audio too short)", 0.0
            
            # Extract embedding from test audio
            embedding = self.embeddings.extract_from_array(audio_processed, sr)
            
            # Search using MAX score strategy
            # This compares against ALL stored embeddings and takes the best match
            results = self.qdrant.search_with_max_score(embedding, limit=1)
            
            if not results:
                logger.info("No matches found in database")
                return False, None, "Unknown Speaker", 0.0
            
            # Get best match
            best_match = results[0]
            score = best_match['score']
            client_id = best_match['client_id']
            
            logger.info(
                f"Best match: {client_id} with score {score:.4f} "
                f"(embedding #{best_match.get('best_embedding_idx', 0)})"
            )
            
            # Check against threshold
            # Lowered from 0.90 to 0.75 for better recall with multi-embedding
            if score >= config.SIMILARITY_THRESHOLD:
                # Match found - retrieve user info
                user = self.mongo.users.find_one({"client_id": client_id})
                name = user['name'] if user else "Unknown"
                
                logger.info(f"✅ Speaker identified: {name} (confidence: {score:.2%})")
                return True, client_id, name, score
            else:
                # Below threshold - unknown speaker
                logger.info(f"⚠️ Score {score:.2%} below threshold {config.SIMILARITY_THRESHOLD:.2%}")
                return False, None, "Unknown Speaker", score
                
        except Exception as e:
            logger.error(f"❌ Error during identification: {e}")
            import traceback
            traceback.print_exc()
            return False, None, "Error during identification", 0.0
    
    def enroll_speaker(self, audio_path: str, name: str):
        """
        Enroll new speaker from audio file.
        
        Args:
            audio_path: Path to audio file
            name: Speaker's name
            
        Returns:
            client_id of enrolled speaker
        """
        try:
            audio, sr = sf.read(audio_path)
            return self.enroll_from_array(audio, name, sr)
        except Exception as e:
            logger.error(f"❌ Error enrolling from file: {e}")
            raise
    
    def enroll_from_array(self, audio: np.ndarray, name: str, sr: int = 16000):
        """
        Enroll new speaker from audio array with multiple embeddings.
        
        Key improvements:
        - Extracts 3-4 embeddings from single audio sample
        - Adaptive count based on audio length
        - Stores all embeddings for robust matching
        
        Args:
            audio: Audio signal as numpy array
            name: Speaker's name
            sr: Sample rate (default: 16000)
            
        Returns:
            client_id of enrolled speaker
            
        Raises:
            ValueError: If audio is too short for enrollment
        """
        try:
            # MINIMAL preprocessing for enrollment
            # We preserve voice characteristics by using light processing
            audio_processed = preprocess_audio(audio, sr, for_enrollment=True)
            
            # Validate minimum length
            min_length = sr * 2.0  # 2 seconds minimum for enrollment
            if len(audio_processed) < min_length:
                logger.warning(
                    f"Audio too short after preprocessing: {len(audio_processed)/sr:.2f}s"
                )
                # Fallback to original with simple normalization
                audio_processed = audio
                max_val = np.abs(audio_processed).max()
                if max_val > 0:
                    audio_processed = audio_processed / max_val
                
                # Still too short? Reject enrollment
                if len(audio_processed) < min_length:
                    raise ValueError(
                        f"Audio too short for enrollment: {len(audio_processed)/sr:.2f}s "
                        f"(minimum: 2.0s)"
                    )
            
            # Generate unique client ID
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            client_id = f"CLIENT_{timestamp}"
            
            # Determine number of embeddings based on audio length
            audio_length_sec = len(audio_processed) / sr
            
            if audio_length_sec < 8:
                num_embeddings = 3  # Short audio: 3 embeddings
            elif audio_length_sec < 20:
                num_embeddings = 4  # Medium audio: 4 embeddings
            else:
                num_embeddings = 5  # Long audio: 5 embeddings
            
            logger.info(
                f"Enrolling {name}: {audio_length_sec:.1f}s audio → "
                f"{num_embeddings} embeddings"
            )
            
            # Extract MULTIPLE embeddings from this single sample
            embeddings_list = self.embeddings.extract_multiple_embeddings(
                audio_processed, sr, num_embeddings=num_embeddings
            )
            
            # Store user metadata in MongoDB
            user_doc = {
                "client_id": client_id,
                "name": name,
                "created_at": datetime.utcnow(),
                "is_sample": False,
                "num_embeddings": len(embeddings_list),
                "audio_length_sec": audio_length_sec
            }
            self.mongo.users.insert_one(user_doc)
            
            # Store ALL embeddings in Qdrant
            self.qdrant.insert_multiple(client_id, embeddings_list)
            
            logger.info(
                f"✅ Enrolled {name} ({client_id}) with "
                f"{len(embeddings_list)} embeddings"
            )
            
            return client_id
            
        except Exception as e:
            logger.error(f"❌ Error during enrollment: {e}")
            raise
    
    def get_all_speakers(self):
        """
        Get all enrolled speakers.
        
        Returns:
            List of user documents (without MongoDB _id field)
        """
        try:
            users = list(self.mongo.users.find({}, {"_id": 0}))
            logger.info(f"Retrieved {len(users)} enrolled speakers")
            return users
        except Exception as e:
            logger.error(f"❌ Error retrieving speakers: {e}")
            return []
    
    def delete_speaker(self, client_id: str):
        """
        Delete speaker from both databases.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Delete from MongoDB
            mongo_deleted = self.mongo.delete_user(client_id)
            
            # Delete ALL embeddings from Qdrant
            self.qdrant.delete_all_embeddings(client_id)
            
            if mongo_deleted:
                logger.info(f"✅ Deleted speaker: {client_id}")
                return True
            else:
                logger.warning(f"⚠️ Speaker not found: {client_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error deleting speaker: {e}")
            return False
    
    def get_stats(self):
        """
        Get system statistics.
        
        Returns:
            Dict with system stats
        """
        try:
            users = self.get_all_speakers()
            qdrant_stats = self.qdrant.get_stats()
            
            total_embeddings = sum(u.get('num_embeddings', 1) for u in users)
            
            return {
                "total_users": len(users),
                "total_embeddings": total_embeddings,
                "avg_embeddings_per_user": total_embeddings / len(users) if users else 0,
                "qdrant_points": qdrant_stats.get('total_points', 0),
                "threshold": config.SIMILARITY_THRESHOLD
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {}