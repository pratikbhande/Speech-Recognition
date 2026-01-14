"""Voice embedding extraction using SpeechBrain and WeSpeaker."""

import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf
import config
import tempfile
import os


class VoiceEmbedder:
    """Extract voice embeddings using SpeechBrain ECAPA-TDNN or WeSpeaker ResNet."""
    
    def __init__(self, model_type=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type or config.MODEL_TYPE
        
        if self.model_type == "speechbrain":
            self._init_speechbrain()
        elif self.model_type == "wespeaker":
            self._init_wespeaker()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _init_speechbrain(self):
        """Initialize SpeechBrain model."""
        print("🔄 Loading SpeechBrain ECAPA-TDNN model...")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        print("✅ SpeechBrain model loaded (ECAPA-TDNN)")
    
    def _init_wespeaker(self):
        """Initialize WeSpeaker runtime model."""
        try:
            from wespeakerruntime.speaker import Speaker
            
            print("🔄 Loading WeSpeaker model...")
            
            # The 'chinese' model works for English (trained on VoxCeleb)
            self.model = Speaker(lang='chinese')
            
            print("✅ WeSpeaker model loaded")
            
        except Exception as e:
            print(f"❌ WeSpeaker failed: {str(e)}")
            print("⚠️ Falling back to SpeechBrain...")
            self.model_type = "speechbrain"
            self._init_speechbrain()
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract embedding from audio file."""
        if self.model_type == "speechbrain":
            return self._extract_speechbrain_file(audio_path)
        else:
            return self._extract_wespeaker_file(audio_path)
    
    def extract_from_array(self, audio_array: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract embedding from audio array."""
        if self.model_type == "speechbrain":
            return self._extract_speechbrain_array(audio_array, sr)
        else:
            return self._extract_wespeaker_array(audio_array, sr)
    
    def _extract_speechbrain_file(self, audio_path: str) -> np.ndarray:
        """Extract using SpeechBrain from file."""
        signal, sr = torchaudio.load(audio_path)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
        
        with torch.no_grad():
            embedding = self.model.encode_batch(signal.to(self.device))
            embedding = embedding.squeeze().cpu().numpy()
        
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _extract_speechbrain_array(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """Extract using SpeechBrain from array."""
        signal = torch.FloatTensor(audio_array).unsqueeze(0)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
        
        with torch.no_grad():
            embedding = self.model.encode_batch(signal.to(self.device))
            embedding = embedding.squeeze().cpu().numpy()
        
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _extract_wespeaker_file(self, audio_path: str) -> np.ndarray:
        """Extract using WeSpeaker runtime from file."""
        embedding = self.model.extract_embedding(audio_path)
        
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        embedding = embedding.squeeze()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _extract_wespeaker_array(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """Extract using WeSpeaker runtime from array."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_array, sr)
        
        try:
            embedding = self.model.extract_embedding(tmp_path)
            
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            embedding = embedding.squeeze()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return embedding
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))