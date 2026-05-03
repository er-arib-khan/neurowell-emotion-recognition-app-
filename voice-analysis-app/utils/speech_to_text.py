"""
Local speech-to-text using Whisper (simplified version)
"""

import numpy as np
import warnings
import os

# Try importing whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("Whisper not installed. Using simple fallback.")

class LocalSTT:
    """Speech-to-text using OpenAI's Whisper (local)"""
    
    def __init__(self, model_size="base"):
        """
        Initialize Whisper for local transcription
        Model sizes: tiny, base, small, medium, large
        """
        self.model_size = model_size
        self.available = WHISPER_AVAILABLE
        
        if self.available:
            print(f"Loading Whisper {model_size} model...")
            try:
                self.model = whisper.load_model(model_size)
                print("✅ Whisper model loaded!")
            except Exception as e:
                print(f"❌ Failed to load Whisper: {e}")
                self.available = False
        else:
            print("⚠️ Whisper not available. Install with: pip install openai-whisper")
    
    def transcribe(self, audio, sample_rate=16000):
        """Transcribe audio to text"""
        if not self.available:
            return "[Whisper not installed]"
        
        try:
            # Ensure audio is in the right format
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Convert to float32 if needed
            audio = audio.astype(np.float32)
            
            # Transcribe
            result = self.model.transcribe(
                audio, 
                fp16=False,
                language='en'
            )
            
            return result['text'].strip()
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            return f"[Error: {str(e)}]"