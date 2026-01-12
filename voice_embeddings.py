"""Voice embedding extraction using SpeechBrain."""

import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from typing import Union
import soundfile as sf


class VoiceEmbedder:
    """Extract voice embeddings using SpeechBrain ECAPA-TDNN."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract embedding from audio file."""
        signal, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(signal.to(self.device))
            embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def extract_from_array(self, audio_array: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract embedding from audio array."""
        signal = torch.FloatTensor(audio_array).unsqueeze(0)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
        
        with torch.no_grad():
            embedding = self.model.encode_batch(signal.to(self.device))
            embedding = embedding.squeeze().cpu().numpy()
        
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))