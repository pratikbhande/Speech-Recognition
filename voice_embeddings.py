"""
Voice embedding extraction using SpeechBrain ECAPA-TDNN.

This module provides voice embedding extraction with support for:
- Single embedding extraction
- Multiple embeddings from one audio sample (for robustness)
- Audio augmentation for variation
- Efficient batch processing
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from typing import Union, List
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class VoiceEmbedder:
    """
    Extract voice embeddings using SpeechBrain ECAPA-TDNN.
    
    Features:
    - Single embedding extraction for standard use
    - Multiple embeddings from one sample for improved robustness
    - Light audio augmentation for variation
    - GPU acceleration when available
    """
    
    def __init__(self):
        """Initialize the ECAPA-TDNN model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing VoiceEmbedder on device: {self.device}")
        
        try:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            logger.info("✅ ECAPA-TDNN model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract single embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Normalized embedding vector (192-dim)
        """
        try:
            signal, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                signal = resampler(signal)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(signal.to(self.device))
                embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding from {audio_path}: {e}")
            raise
    
    def extract_from_array(self, audio_array: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract single embedding from audio array.
        
        Args:
            audio_array: Audio signal as numpy array
            sr: Sample rate (default: 16000)
            
        Returns:
            Normalized embedding vector (192-dim)
        """
        try:
            return self._extract_single(audio_array, sr)
        except Exception as e:
            logger.error(f"Error extracting embedding from array: {e}")
            raise
    
    def extract_multiple_embeddings(
        self, 
        audio_array: np.ndarray, 
        sr: int = 16000, 
        num_embeddings: int = 4
    ) -> List[np.ndarray]:
        """
        Extract multiple embeddings from single audio sample for robustness.
        
        Strategy:
        1. Extract from different temporal segments (if audio is long enough)
        2. Always include full-length embedding
        3. Add augmented versions to reach target count
        
        This creates variation that helps match the same speaker under
        different conditions (noise, mood, health, recording quality).
        
        Args:
            audio_array: Audio signal as numpy array
            sr: Sample rate (default: 16000)
            num_embeddings: Number of embeddings to create (default: 4)
            
        Returns:
            List of normalized embedding vectors
        """
        embeddings = []
        audio_length = len(audio_array)
        min_segment_length = sr * 3  # 3 seconds minimum per segment
        
        try:
            # Strategy 1: Extract from different temporal segments
            # This captures natural variation in speech over time
            if audio_length >= min_segment_length * 2:
                num_segments = min(num_embeddings - 1, int(audio_length / min_segment_length))
                
                # Distribute segments evenly across the audio
                segment_starts = np.linspace(
                    0, 
                    audio_length - min_segment_length, 
                    num_segments, 
                    dtype=int
                )
                
                for idx, start in enumerate(segment_starts):
                    segment = audio_array[start:start + min_segment_length]
                    emb = self._extract_single(segment, sr)
                    embeddings.append(emb)
                    logger.debug(f"Extracted embedding from segment {idx+1}/{num_segments}")
            
            # Strategy 2: Always include full audio embedding
            # This captures the complete voice signature
            full_emb = self._extract_single(audio_array, sr)
            embeddings.append(full_emb)
            logger.debug("Extracted full-length embedding")
            
            # Strategy 3: Add augmented versions to reach target count
            # This simulates different recording conditions
            augmentation_count = 0
            while len(embeddings) < num_embeddings:
                aug_audio = self._augment_audio(audio_array, sr)
                aug_emb = self._extract_single(aug_audio, sr)
                embeddings.append(aug_emb)
                augmentation_count += 1
            
            if augmentation_count > 0:
                logger.debug(f"Added {augmentation_count} augmented embeddings")
            
            logger.info(f"✅ Created {len(embeddings)} embeddings from single sample")
            return embeddings[:num_embeddings]
            
        except Exception as e:
            logger.error(f"Error creating multiple embeddings: {e}")
            # Fallback: return at least one embedding
            return [self._extract_single(audio_array, sr)]
    
    def _extract_single(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """
        Internal method to extract single embedding.
        
        Args:
            audio_array: Audio signal
            sr: Sample rate
            
        Returns:
            Normalized embedding vector
        """
        signal = torch.FloatTensor(audio_array).unsqueeze(0)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            signal = resampler(signal)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(signal.to(self.device))
            embedding = embedding.squeeze().cpu().numpy()
        
        # L2 normalization
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _augment_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply light augmentation to create audio variation.
        
        Randomly selects ONE augmentation type:
        - Gaussian noise: Simulates recording noise
        - Pitch shift: Simulates voice variation
        - Speed variation: Simulates speech rate changes
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Augmented audio signal
        """
        aug_type = np.random.choice(['noise', 'pitch', 'speed'])
        
        try:
            if aug_type == 'noise':
                # Add very light Gaussian noise (SNR ~40dB)
                noise_level = 0.002  # Very subtle
                noise = np.random.normal(0, noise_level, len(audio))
                augmented = audio + noise
                # Ensure we don't clip
                augmented = np.clip(augmented, -1.0, 1.0)
                return augmented
            
            elif aug_type == 'pitch':
                # Slight pitch shift (±2 semitones)
                # Simulates voice variation due to mood/health
                signal = torch.FloatTensor(audio).unsqueeze(0)
                pitch_shift = np.random.choice([-2, -1, 1, 2])
                shifter = T.PitchShift(sr, n_steps=pitch_shift)
                shifted = shifter(signal).squeeze().numpy()
                return shifted
            
            else:  # speed
                # Slight speed change (97-103%)
                # Simulates different speech rates
                signal = torch.FloatTensor(audio).unsqueeze(0)
                speed_factor = np.random.uniform(0.97, 1.03)
                
                # Change speed by resampling
                temp_sr = int(sr * speed_factor)
                resampler = T.Resample(sr, temp_sr)
                sped = resampler(signal)
                
                # Resample back to original rate
                resampler_back = T.Resample(temp_sr, sr)
                result = resampler_back(sped).squeeze().numpy()
                
                # Trim or pad to original length
                if len(result) > len(audio):
                    result = result[:len(audio)]
                elif len(result) < len(audio):
                    result = np.pad(result, (0, len(audio) - len(result)), mode='constant')
                
                return result
                
        except Exception as e:
            logger.warning(f"Augmentation failed ({aug_type}), returning original: {e}")
            return audio
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Similarity score (0 to 1, higher is more similar)
        """
        return float(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )