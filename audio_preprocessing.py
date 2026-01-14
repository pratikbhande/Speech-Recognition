"""
Lightweight audio preprocessing for speaker recognition.
MINIMAL processing to preserve voice characteristics.
"""

import numpy as np
import librosa
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("⚠️ Install noisereduce for better results: pip install noisereduce")


class AudioPreprocessor:
    """Lightweight preprocessing - preserves voice characteristics."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def process(self, audio, sr=None, mode='light'):
        """
        Minimal preprocessing pipeline.
        
        Args:
            audio: numpy array (float32)
            sr: sample rate
            mode: 'light' for enrollment, 'standard' for identification
            
        Returns:
            preprocessed audio (numpy array)
        """
        if sr is None:
            sr = self.sample_rate
            
        # Step 1: Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Step 2: Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Step 3: Remove DC offset
        audio = audio - np.mean(audio)
        
        # Step 4: Normalize amplitude
        audio = self._normalize(audio)
        
        # Step 5: Light noise reduction ONLY if mode is standard
        if mode == 'standard' and HAS_NOISEREDUCE and len(audio) > sr * 0.5:
            audio = self._reduce_noise_light(audio, sr)
        
        # Step 6: Final normalization
        audio = self._normalize(audio)
        
        return audio
    
    def _normalize(self, audio):
        """Normalize audio to [-1, 1] range."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _reduce_noise_light(self, audio, sr):
        """LIGHT noise reduction - preserves voice characteristics."""
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,
                prop_decrease=0.5,  # Only 50% reduction (was 1.0 = 100%)
                freq_mask_smooth_hz=1000,
                time_mask_smooth_ms=100
            )
            return reduced
        except Exception as e:
            print(f"⚠️ Noise reduction skipped: {e}")
            return audio


# Global preprocessor instance
_preprocessor = None

def get_preprocessor():
    """Get or create global preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor()
    return _preprocessor


def preprocess_audio(audio, sr=16000, for_enrollment=False):
    """
    Convenience function for preprocessing audio.
    
    Args:
        audio: numpy array
        sr: sample rate
        for_enrollment: if True, use lighter processing (preserves voice)
        
    Returns:
        preprocessed audio
    """
    preprocessor = get_preprocessor()
    
    if for_enrollment:
        # LIGHT processing for enrollment - preserve voice characteristics
        return preprocessor.process(audio, sr, mode='light')
    else:
        # STANDARD processing for identification - light noise reduction
        return preprocessor.process(audio, sr, mode='standard')