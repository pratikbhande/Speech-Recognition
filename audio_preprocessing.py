"""
Real-time audio preprocessing for robust speaker recognition.
Handles: noise, voice variations, background sounds, recordings.
"""

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("⚠️ Install noisereduce for better results: pip install noisereduce")


class AudioPreprocessor:
    """Real-time audio preprocessing pipeline."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Speech frequency range (human voice)
        self.low_freq = 80   # Hz
        self.high_freq = 7500  # Hz (below Nyquist for 16kHz)
        
        # VAD parameters
        self.vad_threshold = 0.02  # Energy threshold
        self.vad_frame_length = 0.025  # 25ms frames
        
    def process(self, audio, sr=None):
        """
        Main preprocessing pipeline.
        
        Args:
            audio: numpy array (float32)
            sr: sample rate (if different from 16kHz)
            
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
        
        # Step 3: Normalize amplitude
        audio = self._normalize(audio)
        
        # Step 4: Remove DC offset
        audio = audio - np.mean(audio)
        
        # Step 5: Noise reduction (if available)
        if HAS_NOISEREDUCE:
            audio = self._reduce_noise(audio, sr)
        
        # Step 6: Bandpass filter (focus on speech frequencies)
        audio = self._bandpass_filter(audio, sr)
        
        # Step 7: Voice Activity Detection (remove silence)
        audio = self._apply_vad(audio, sr)
        
        # Step 8: Final normalization
        audio = self._normalize(audio)
        
        # Step 9: Pre-emphasis (reduce low-frequency noise)
        audio = self._preemphasis(audio, coeff=0.97)
        
        return audio
    
    def _normalize(self, audio):
        """Normalize audio to [-1, 1] range."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _reduce_noise(self, audio, sr):
        """Remove background noise using spectral gating."""
        try:
            # Reduce noise
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,  # Non-stationary noise
                prop_decrease=1.0,  # Aggressive reduction
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50
            )
            return reduced
        except Exception as e:
            print(f"⚠️ Noise reduction skipped: {e}")
            return audio
    
    def _bandpass_filter(self, audio, sr):
        """Apply bandpass filter to focus on speech frequencies (80Hz - 7.5kHz)."""
        nyquist = sr / 2.0
        
        # Ensure frequencies are within valid range (0 < Wn < 1)
        low = max(self.low_freq / nyquist, 0.001)  # Minimum 0.001
        high = min(self.high_freq / nyquist, 0.99)  # Maximum 0.99
        
        # Validate range
        if low >= high:
            print("⚠️ Bandpass filter skipped: invalid frequency range")
            return audio
        
        try:
            # Design Butterworth bandpass filter (4th order)
            b, a = signal.butter(4, [low, high], btype='band')
            
            # Apply filter (zero-phase filtering)
            filtered = signal.filtfilt(b, a, audio)
            
            return filtered
        except Exception as e:
            print(f"⚠️ Bandpass filter skipped: {e}")
            return audio
    
    def _apply_vad(self, audio, sr):
        """
        Voice Activity Detection - remove silence/noise-only segments.
        Keeps only parts with speech energy.
        """
        # Frame parameters
        frame_length = int(self.vad_frame_length * sr)
        hop_length = frame_length // 2
        
        # Need minimum length
        if len(audio) < frame_length:
            return audio
        
        # Calculate short-time energy
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio) - frame_length, hop_length)
        ])
        
        # Normalize energy
        if energy.max() > 0:
            energy = energy / energy.max()
        
        # Find speech segments (above threshold)
        speech_frames = energy > self.vad_threshold
        
        # If too few speech frames, return original
        if speech_frames.sum() < 5:
            return audio
        
        # Expand frames back to audio samples
        speech_mask = np.repeat(speech_frames, hop_length)
        
        # Ensure mask matches audio length
        if len(speech_mask) < len(audio):
            speech_mask = np.pad(speech_mask, (0, len(audio) - len(speech_mask)), constant_values=False)
        elif len(speech_mask) > len(audio):
            speech_mask = speech_mask[:len(audio)]
        
        # Keep only speech segments
        speech_audio = audio[speech_mask]
        
        # If too much was removed, return original (safety check)
        if len(speech_audio) < sr * 0.5:  # Less than 0.5 seconds
            return audio
        
        return speech_audio
    
    def _preemphasis(self, audio, coeff=0.97):
        """
        Apply pre-emphasis filter to balance frequency spectrum.
        Reduces low-frequency noise, enhances high frequencies.
        """
        emphasized = np.append(audio[0], audio[1:] - coeff * audio[:-1])
        return emphasized
    
    def enhance_for_recognition(self, audio, sr=None):
        """
        Special enhancement for speaker recognition.
        Focuses on voice characteristics, reduces environmental factors.
        """
        if sr is None:
            sr = self.sample_rate
        
        # Basic preprocessing
        audio = self.process(audio, sr)
        
        # Additional enhancement: median filtering to remove impulse noise
        audio = median_filter(audio, size=3)
        
        # Spectral smoothing (reduces recording artifacts)
        audio = self._spectral_smoothing(audio, sr)
        
        return audio
    
    def _spectral_smoothing(self, audio, sr):
        """Smooth spectrum to reduce recording artifacts."""
        try:
            # Short-time Fourier transform
            stft = librosa.stft(audio, n_fft=512, hop_length=256)
            
            # Magnitude and phase
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Smooth magnitude spectrum
            from scipy.ndimage import gaussian_filter
            smoothed_mag = gaussian_filter(magnitude, sigma=1.0)
            
            # Reconstruct
            smoothed_stft = smoothed_mag * np.exp(1j * phase)
            audio_smoothed = librosa.istft(smoothed_stft, hop_length=256, length=len(audio))
            
            return audio_smoothed
        except Exception as e:
            print(f"⚠️ Spectral smoothing skipped: {e}")
            return audio


# Global preprocessor instance
_preprocessor = None

def get_preprocessor():
    """Get or create global preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor()
    return _preprocessor


def preprocess_audio(audio, sr=16000, enhance=True):
    """
    Convenience function for preprocessing audio.
    
    Args:
        audio: numpy array
        sr: sample rate
        enhance: if True, apply extra enhancement for recognition
        
    Returns:
        preprocessed audio
    """
    preprocessor = get_preprocessor()
    
    if enhance:
        return preprocessor.enhance_for_recognition(audio, sr)
    else:
        return preprocessor.process(audio, sr)