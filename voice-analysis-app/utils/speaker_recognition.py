"""
Speaker recognition using ECAPA-TDNN
"""

import numpy as np
import librosa
import warnings
import torch

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Try to import speechbrain with proper error handling
try:
    # Set torchaudio backend before importing speechbrain
    import torchaudio
    # Try to set backend if available
    try:
        if hasattr(torchaudio, 'set_audio_backend'):
            available_backends = torchaudio.list_audio_backends() if hasattr(torchaudio, 'list_audio_backends') else []
            if 'soundfile' in available_backends:
                torchaudio.set_audio_backend('soundfile')
            elif 'sox_io' in available_backends:
                torchaudio.set_audio_backend('sox_io')
    except:
        pass
    
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError as e:
    SPEECHBRAIN_AVAILABLE = False
    warnings.warn(f"SpeechBrain not available: {e}. Speaker recognition will be disabled.")
except Exception as e:
    SPEECHBRAIN_AVAILABLE = False
    warnings.warn(f"Error loading SpeechBrain: {e}. Speaker recognition will be disabled.")

class SpeakerRecognizer:
    """Speaker identification and verification"""
    
    def __init__(self):
        if not SPEECHBRAIN_AVAILABLE:
            print("⚠️ SpeechBrain not installed or incompatible. Speaker recognition unavailable.")
            print("To install: pip install speechbrain")
            self.available = False
            return
            
        self.available = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained speaker embedding model
        print(f"Loading speaker recognition model on {self.device}...")
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)}
            )
            print("✅ Speaker model loaded!")
        except Exception as e:
            print(f"❌ Failed to load speaker model: {e}")
            self.available = False
            return
        
        # Known speakers database
        self.speaker_embeddings = {}
        self.speaker_names = {}
        self.next_speaker_id = 0
    
    def extract_embedding(self, audio, sample_rate=16000):
        """Extract speaker embedding from audio"""
        if not self.available:
            return None
            
        try:
            # Preprocess
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            # Extract embedding
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(audio_tensor)
            
            return embeddings.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def register_speaker(self, audio, sample_rate=16000, name=None):
        """Register a new speaker"""
        if not self.available:
            return None
            
        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            return None
        
        speaker_id = self.next_speaker_id
        self.speaker_embeddings[speaker_id] = embedding
        self.speaker_names[speaker_id] = name or f"Speaker_{speaker_id}"
        self.next_speaker_id += 1
        
        return speaker_id
    
    def identify(self, audio, sample_rate=16000, threshold=0.7):
        """Identify speaker from audio"""
        if not self.available or not self.speaker_embeddings:
            return "Unknown (speaker recognition unavailable)", 0.0
        
        # Extract embedding for unknown audio
        unknown_embedding = self.extract_embedding(audio, sample_rate)
        if unknown_embedding is None:
            return "Error extracting embedding", 0.0
        
        # Compare with known speakers
        best_match = None
        best_similarity = -1
        
        for speaker_id, known_embedding in self.speaker_embeddings.items():
            # Cosine similarity
            similarity = np.dot(unknown_embedding, known_embedding) / (
                np.linalg.norm(unknown_embedding) * np.linalg.norm(known_embedding) + 1e-8
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id
        
        # Check against threshold
        if best_similarity >= threshold and best_match is not None:
            return self.speaker_names[best_match], best_similarity
        else:
            return "Unknown", best_similarity
    
    def verify(self, audio, speaker_id, sample_rate=16000, threshold=0.7):
        """Verify if audio matches a specific speaker"""
        if not self.available or speaker_id not in self.speaker_embeddings:
            return False, 0.0
        
        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            return False, 0.0
            
        known_embedding = self.speaker_embeddings[speaker_id]
        
        similarity = np.dot(embedding, known_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(known_embedding) + 1e-8
        )
        
        return similarity >= threshold, similarity