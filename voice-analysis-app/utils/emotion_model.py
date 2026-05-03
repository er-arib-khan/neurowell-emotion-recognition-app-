"""
Emotion detection module with SimpleEmotionDetector
"""

import numpy as np
import librosa
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class SimpleEmotionDetector:
    """Simple rule-based emotion detection using audio features"""
    
    def __init__(self):
        self.available = True
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'calm', 'fear', 'disgust', 'surprise']
        print("✅ Simple rule-based emotion detector loaded!")
    
    def extract_features(self, audio, sr):
        """Extract audio features for emotion detection"""
        try:
            # Ensure audio is not empty
            if len(audio) == 0:
                return None
                
            # Energy (loudness)
            energy = np.mean(audio ** 2)
            
            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            # Get non-zero pitches
            pitches_nonzero = pitches[pitches > 0]
            pitch = np.mean(pitches_nonzero) if len(pitches_nonzero) > 0 else 0
            
            # Zero crossing rate (measure of noise/roughness)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_centroid = np.mean(spectral_centroids)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)
            rms_mean = np.mean(rms)
            
            # Tempo (speech rate) - use a default if can't detect
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                tempo = tempo if tempo else 120
            except:
                tempo = 120
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            rolloff_mean = np.mean(spectral_rolloff)
            
            # MFCCs (for more detailed analysis)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            return {
                'energy': float(energy),
                'pitch': float(pitch),
                'zcr': float(np.mean(zcr)),
                'spectral_centroid': float(spectral_centroid),
                'rms': float(rms_mean),
                'tempo': float(tempo),
                'rolloff': float(rolloff_mean),
                'mfcc_mean': float(np.mean(mfcc_means)),
                'mfcc_std': float(np.std(mfcc_means))
            }
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio, sample_rate=16000):
        """Predict emotion based on audio features"""
        try:
            features = self.extract_features(audio, sample_rate)
            if features is None:
                return "neutral", 0.5
            
            # Rule-based classification with multiple features
            # High energy + high pitch + high zcr = happy/excited
            if features['energy'] > 0.05 and features['pitch'] > 180:
                if features['zcr'] > 0.07:
                    emotion = "happy"
                    confidence = min(0.85, features['energy'] * 15)
                else:
                    emotion = "surprise"
                    confidence = min(0.8, features['energy'] * 12)
            
            # High energy + high pitch + low zcr = angry
            elif features['energy'] > 0.04 and features['pitch'] > 150:
                if features['zcr'] < 0.06:
                    emotion = "angry"
                    confidence = min(0.9, features['energy'] * 18)
                else:
                    emotion = "fear"
                    confidence = min(0.75, features['energy'] * 10)
            
            # Low energy + low pitch = sad
            elif features['energy'] < 0.02 and features['pitch'] < 120:
                if features['tempo'] < 100:
                    emotion = "sad"
                    confidence = 0.7
                else:
                    emotion = "calm"
                    confidence = 0.8
            
            # Low energy + medium pitch = calm/neutral
            elif features['energy'] < 0.03:
                if features['spectral_centroid'] < 2000:
                    emotion = "calm"
                    confidence = 0.75
                else:
                    emotion = "neutral"
                    confidence = 0.65
            
            # Default case
            else:
                emotion = "neutral"
                confidence = 0.6
            
            # Adjust confidence based on feature stability
            if features['mfcc_std'] < 10:
                confidence = min(confidence + 0.1, 0.95)
            
            return emotion, confidence
            
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return "neutral", 0.5
    
    def predict_batch(self, audio_batch, sample_rates):
        """Predict emotions for multiple audio samples"""
        results = []
        for audio, sr in zip(audio_batch, sample_rates):
            emotion, conf = self.predict(audio, sr)
            results.append((emotion, conf))
        return results

# Also keep the original EmotionDetector class if needed
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class EmotionDetector:
    """Advanced emotion detection using transformers (if available)"""
    
    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available, using SimpleEmotionDetector instead")
            self.simple_detector = SimpleEmotionDetector()
            self.available = True
            return
            
        self.available = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use a reliable model
        self.model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        
        # Emotion labels
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']
        
        print(f"Loading emotion model on {self.device}...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self.model.eval()
            print("✅ Advanced emotion model loaded!")
        except Exception as e:
            print(f"❌ Failed to load advanced model: {e}")
            print("Using SimpleEmotionDetector as fallback")
            self.simple_detector = SimpleEmotionDetector()
    
    def preprocess_audio(self, audio, target_sr=16000):
        """Preprocess audio for the model"""
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            if target_sr != 16000:
                audio = librosa.resample(audio, orig_sr=target_sr, target_sr=16000)
            
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))
            
            return audio
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def predict(self, audio, sample_rate=16000):
        """Predict emotion from audio"""
        # If advanced model failed, use simple detector
        if not hasattr(self, 'model'):
            return self.simple_detector.predict(audio, sample_rate)
        
        try:
            audio = self.preprocess_audio(audio, sample_rate)
            if audio is None:
                return "neutral", 0.5
            
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = F.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            if predicted_class < len(self.emotions):
                emotion = self.emotions[predicted_class]
            else:
                emotion = "neutral"
            
            return emotion, confidence
            
        except Exception as e:
            print(f"Error in advanced emotion prediction: {e}")
            # Fallback to simple detector
            return self.simple_detector.predict(audio, sample_rate)