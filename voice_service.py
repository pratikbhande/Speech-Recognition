"""Voice recognition service with optimized preprocessing."""

from pathlib import Path
import numpy as np
import soundfile as sf
import config
from voice_embeddings import VoiceEmbedder
from database import MongoManager, QdrantManager
from datetime import datetime
from audio_preprocessing import preprocess_audio


class VoiceService:
    """Main service for voice recognition operations."""
    
    def __init__(self):
        self.embeddings = VoiceEmbedder()
        self.mongo = MongoManager()
        self.qdrant = QdrantManager()
        
    def identify_speaker(self, audio_path: str):
        """Identify speaker from audio file."""
        audio, sr = sf.read(audio_path)
        return self.identify_from_array(audio, sr)
    
    def identify_from_array(self, audio: np.ndarray, sr: int = 16000):
        """
        Identify speaker from audio array with preprocessing.
        
        Returns:
            (is_known, client_id, name, confidence)
        """
        # LIGHT preprocessing for identification
        audio_processed = preprocess_audio(audio, sr, for_enrollment=False)
        
        # Ensure minimum length (1 second)
        if len(audio_processed) < sr * 1.0:
            print(f"⚠️ Audio too short after preprocessing: {len(audio_processed)/sr:.2f}s")
            # Use original if preprocessing destroyed too much
            audio_processed = audio
        
        # Extract embedding
        embedding = self.embeddings.extract_from_array(audio_processed, sr)
        
        # Search in vector DB
        results = self.qdrant.search(embedding, limit=1)
        
        if not results:
            return False, None, "Unknown Speaker", 0.0
        
        best_match = results[0]
        score = best_match.score
        
        if score >= config.SIMILARITY_THRESHOLD:
            # Found match
            client_id = best_match.payload['client_id']
            user = self.mongo.users.find_one({"client_id": client_id})
            name = user['name'] if user else "Unknown"
            return True, client_id, name, score
        else:
            # Below threshold
            return False, None, "Unknown Speaker", score
    
    def enroll_speaker(self, audio_path: str, name: str):
        """Enroll new speaker from audio file."""
        audio, sr = sf.read(audio_path)
        return self.enroll_from_array(audio, name, sr)
    
    def enroll_from_array(self, audio: np.ndarray, name: str, sr: int = 16000):
        """
        Enroll new speaker from audio array with MINIMAL preprocessing.
        
        Returns:
            client_id
        """
        # MINIMAL preprocessing for enrollment - preserve voice characteristics
        audio_processed = preprocess_audio(audio, sr, for_enrollment=True)
        
        # Ensure we have enough audio
        if len(audio_processed) < sr * 1.0:
            print(f"⚠️ Audio too short after preprocessing: {len(audio_processed)/sr:.2f}s, using original")
            # Use original audio if preprocessing removed too much
            audio_processed = audio
            # Just normalize the original
            max_val = np.abs(audio_processed).max()
            if max_val > 0:
                audio_processed = audio_processed / max_val
        
        # Generate client ID
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        client_id = f"CLIENT_{timestamp}"
        
        # Extract embedding
        embedding = self.embeddings.extract_from_array(audio_processed, sr)
        
        # Store in MongoDB
        user_doc = {
            "client_id": client_id,
            "name": name,
            "created_at": datetime.utcnow(),
            "is_sample": False
        }
        self.mongo.users.insert_one(user_doc)
        
        # Store in Qdrant
        point_id = hash(client_id) & 0x7FFFFFFF
        self.qdrant.insert(point_id, embedding, {"client_id": client_id})
        
        return client_id
    
    def get_all_speakers(self):
        """Get all enrolled speakers."""
        return list(self.mongo.users.find({}, {"_id": 0}))
    
    def delete_speaker(self, client_id: str):
        """Delete speaker from both databases."""
        self.mongo.users.delete_one({"client_id": client_id})
        point_id = hash(client_id) & 0x7FFFFFFF
        self.qdrant.client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=[point_id]
        )