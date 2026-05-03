"""
Audio processing utilities for voice analysis
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import warnings

# Try to import noisereduce, but provide fallback if not available
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    warnings.warn("noisereduce not installed. Noise reduction will be disabled.")

class AudioProcessor:
    """Handle audio preprocessing and feature extraction"""
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        
    def preprocess(self, audio, sr):
        """Preprocess audio for model input"""
        # Resample if necessary
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Apply noise reduction if available
        if NOISEREDUCE_AVAILABLE:
            try:
                audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.5)
            except Exception as e:
                warnings.warn(f"Noise reduction failed: {e}")
        
        # Normalize
        if np.max(np.abs(audio)) > 0:  # Avoid division by zero
            audio = audio / np.max(np.abs(audio))
        
        return audio, sr
    
    def extract_mfcc(self, audio, sr, n_mfcc=40):
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        return np.mean(mfccs.T, axis=0)
    
    def extract_chroma(self, audio, sr):
        """Extract chromagram features"""
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        return np.mean(chroma.T, axis=0)
    
    def extract_spectral(self, audio, sr):
        """Extract spectral features"""
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        
        return np.array([
            np.mean(spectral_centroids),
            np.mean(spectral_rolloff),
            np.mean(spectral_bandwidth)
        ])
    
    def extract_prosody(self, audio, sr):
        """Extract prosodic features (pitch, energy)"""
        # Pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Energy
        energy = np.sum(audio ** 2) / len(audio) if len(audio) > 0 else 0
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        return np.array([pitch, energy, np.mean(zcr)])

def extract_audio_features(audio, sr):
    """Extract all features from audio"""
    processor = AudioProcessor()
    audio, sr = processor.preprocess(audio, sr)
    
    features = []
    
    # Extract different feature types
    features.extend(processor.extract_mfcc(audio, sr))
    features.extend(processor.extract_chroma(audio, sr))
    features.extend(processor.extract_spectral(audio, sr))
    features.extend(processor.extract_prosody(audio, sr))
    
    return np.array(features)