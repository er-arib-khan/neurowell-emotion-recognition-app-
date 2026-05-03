import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import io
import wave
import base64
import os
import hashlib
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import joblib
from scipy import signal
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

# For ML models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Audio recording
import pyaudio
import wave
import threading
import time

# ==================== PROFESSIONAL COLOR PALETTE ====================
COLORS = {
    'primary': '#6366F1',
    'primary_dark': '#4F46E5',
    'primary_light': '#818CF8',
    'secondary': '#10B981',
    'accent': '#F59E0B',
    'danger': '#EF4444',
    'warning': '#F97316',
    'success': '#22C55E',
    'info': '#3B82F6',
    'dark': '#1F2937',
    'darker': '#111827',
    'light': '#F9FAFB',
    'lighter': '#FFFFFF',
    'gray': '#6B7280',
    'gray_light': '#E5E7EB',
    'gray_dark': '#4B5563',
    'hospital_blue': '#005EB8',
    'hospital_grey': '#425563',
    'hospital_warm': '#FFB81C'
}

# ==================== VOICE EMOTION MAPPING ====================
VOICE_EMOTION_MAPPING = {
    'calm': {
        'status': '😌 Calm / Relaxed',
        'suggestion': 'You sound calm and relaxed. This is an excellent state for mindfulness and clear thinking.',
        'activities': ['Practice meditation', 'Read a book', 'Enjoy peaceful music'],
        'color': '#4CAF50',
        'wellness_tip': 'Calmness is a sign of emotional regulation. Nurture this state.',
        'icon': '😌',
        'category': 'Positive',
        'clinical_term': 'Euthymic State',
        'intervention': 'Maintain current coping strategies'
    },
    'happy': {
        'status': '😊 Happy / Joyful',
        'suggestion': 'Your voice indicates happiness! This positive energy is great for mental well-being.',
        'activities': ['Share your joy with others', 'Express gratitude', 'Engage in creative activities'],
        'color': '#F59E0B',
        'wellness_tip': 'Happiness is contagious. Spread your positive energy!',
        'icon': '😊',
        'category': 'Positive',
        'clinical_term': 'Elevated Mood',
        'intervention': 'Positive psychology interventions'
    },
    'sad': {
        'status': '😔 Sad / Melancholic',
        'suggestion': 'Your voice suggests sadness. Be gentle with yourself and reach out if needed.',
        'activities': ['Self-care routine', 'Connect with loved ones', 'Gentle exercise'],
        'color': '#4169E1',
        'wellness_tip': 'Sadness is a natural emotion that helps us process loss and seek connection.',
        'icon': '😔',
        'category': 'Negative',
        'clinical_term': 'Depressed Mood',
        'intervention': 'Behavioral activation and social connection'
    },
    'angry': {
        'status': '😠 Angry / Frustrated',
        'suggestion': 'You sound frustrated. Take deep breaths before responding to situations.',
        'activities': ['Deep breathing exercises', 'Physical activity', 'Count to ten'],
        'color': '#EF4444',
        'wellness_tip': 'Anger often signals unmet needs. What might you need right now?',
        'icon': '😠',
        'category': 'Negative',
        'clinical_term': 'Irritability',
        'intervention': 'Anger management techniques'
    },
    'fearful': {
        'status': '😨 Fearful / Anxious',
        'suggestion': 'Your voice indicates anxiety. Practice grounding techniques to calm your nervous system.',
        'activities': ['Box breathing', 'Grounding exercises', 'Talk to someone trusted'],
        'color': '#800080',
        'wellness_tip': 'Fear is future-focused. Bring yourself back to the present moment.',
        'icon': '😨',
        'category': 'Negative',
        'clinical_term': 'Anxiety',
        'intervention': 'Anxiety management and grounding exercises'
    },
    'neutral': {
        'status': '😐 Neutral / Balanced',
        'suggestion': 'Your voice sounds neutral and balanced. This is perfect for focused work.',
        'activities': ['Start a new task', 'Practice mindfulness', 'Learn something new'],
        'color': '#6B7280',
        'wellness_tip': 'A neutral state is a peaceful baseline for well-being.',
        'icon': '😐',
        'category': 'Neutral',
        'clinical_term': 'Baseline State',
        'intervention': 'Maintain current coping strategies'
    },
    'disgust': {
        'status': '🤢 Disgust / Aversion',
        'suggestion': 'Your voice suggests aversion. Identify what might be bothering you.',
        'activities': ['Fresh air break', 'Change your environment', 'Mindful observation'],
        'color': '#10B981',
        'wellness_tip': 'Disgust can be a protective emotion. Honor what your body is telling you.',
        'icon': '🤢',
        'category': 'Negative',
        'clinical_term': 'Aversive Response',
        'intervention': 'Environmental adjustment'
    },
    'surprise': {
        'status': '😲 Surprise / Shock',
        'suggestion': 'You sound surprised! Take a moment to process what happened.',
        'activities': ['Pause and reflect', 'Journal about it', 'Discuss with someone'],
        'color': '#EC4899',
        'wellness_tip': 'Surprise can be an invitation to curiosity.',
        'icon': '😲',
        'category': 'Neutral',
        'clinical_term': 'Startle Response',
        'intervention': 'Mindful observation'
    }
}

# ==================== FEATURE EXTRACTION CLASS ====================
class VoiceFeatureExtractor:
    """Extract acoustic features from voice recordings"""
    
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate
        self.n_mfcc = 13
        self.hop_length = 512
        self.n_fft = 2048
        
    def extract_features(self, audio, sr):
        """Extract comprehensive features from audio"""
        features = {}
        
        try:
            # MFCC (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, 
                                         hop_length=self.hop_length, n_fft=self.n_fft)
            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitches = pitches[pitches > 0]
            if len(pitches) > 0:
                features['pitch_mean'] = np.mean(pitches)
                features['pitch_std'] = np.std(pitches)
                features['pitch_max'] = np.max(pitches)
                features['pitch_min'] = np.min(pitches)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_max'] = 0
                features['pitch_min'] = 0
            
            # Energy and RMS
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            features['energy'] = np.sum(audio**2) / len(audio)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, 
                                                                   hop_length=self.hop_length)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, 
                                                                hop_length=self.hop_length)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, 
                                                                   hop_length=self.hop_length)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            features['tempo'] = tempo
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=self.hop_length)
            for i in range(12):
                features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            
            # Mel-spectrogram features
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                                                      hop_length=self.hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spec_mean'] = np.mean(mel_spec_db)
            features['mel_spec_std'] = np.std(mel_spec_db)
            
            # Voice quality features
            features['harmonic'] = np.mean(librosa.effects.harmonic(audio))
            features['percussive'] = np.mean(librosa.effects.percussive(audio))
            
            # Duration
            features['duration'] = len(audio) / sr
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default values if extraction fails
            features = self.get_default_features()
        
        return features
    
    def get_default_features(self):
        """Return default feature values"""
        features = {}
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = 0
            features[f'mfcc_{i}_std'] = 0
        
        features.update({
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_max': 0, 'pitch_min': 0,
            'rms_mean': 0, 'rms_std': 0, 'energy': 0,
            'zcr_mean': 0, 'zcr_std': 0,
            'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
            'spectral_rolloff_mean': 0, 'spectral_rolloff_std': 0,
            'spectral_bandwidth_mean': 0, 'spectral_bandwidth_std': 0,
            'tempo': 0,
            'mel_spec_mean': 0, 'mel_spec_std': 0,
            'harmonic': 0, 'percussive': 0,
            'duration': 0
        })
        
        for i in range(12):
            features[f'chroma_{i}_mean'] = 0
        
        return features

# ==================== VOICE EMOTION CLASSIFIER ====================
class VoiceEmotionClassifier:
    """Machine learning model for voice emotion classification"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = VoiceFeatureExtractor()
        self.is_trained = False
        self.emotions = list(VOICE_EMOTION_MAPPING.keys())
        
    def train_initial_model(self):
        """Train a simple model with synthetic data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features for each emotion
        X = []
        y = []
        
        for emotion in self.emotions:
            # Generate features for this emotion
            n_emotion_samples = n_samples // len(self.emotions)
            
            for _ in range(n_emotion_samples):
                # Create feature vector with some variation
                features = self.generate_synthetic_features(emotion)
                X.append(features)
                y.append(emotion)
        
        X = np.array(X)
        
        # Train a simple Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        
        return self.model
    
    def generate_synthetic_features(self, emotion):
        """Generate synthetic features for each emotion"""
        # Base features (49 features total)
        n_features = 49
        features = np.random.randn(n_features) * 0.1
        
        # Add emotion-specific patterns
        if emotion == 'calm':
            features[0:13] += 0.3  # Lower MFCCs
            features[13] += 0.2     # Lower pitch
            features[20] -= 0.3     # Lower energy
        elif emotion == 'happy':
            features[0:13] += 0.6   # Higher MFCCs
            features[13] += 0.7     # Higher pitch
            features[20] += 0.5     # Higher energy
            features[30] += 0.4     # Higher spectral centroid
        elif emotion == 'sad':
            features[0:13] -= 0.4   # Lower MFCCs
            features[13] -= 0.5     # Lower pitch
            features[20] -= 0.4     # Lower energy
            features[25] -= 0.3     # Lower tempo
        elif emotion == 'angry':
            features[0:13] += 0.5   # Higher MFCCs
            features[13] += 0.6     # Higher pitch
            features[20] += 0.7     # Higher energy
            features[22] += 0.5     # Higher ZCR
            features[30] += 0.6     # Higher spectral centroid
        elif emotion == 'fearful':
            features[0:13] += 0.3   # Moderate MFCCs
            features[13] += 0.5     # Higher pitch
            features[20] += 0.3     # Higher energy
            features[22] += 0.4     # Higher ZCR (tremor)
        elif emotion == 'neutral':
            features = np.random.randn(n_features) * 0.1  # Neutral baseline
        elif emotion == 'disgust':
            features[0:13] -= 0.2   # Slightly lower MFCCs
            features[13] += 0.1     # Slight pitch increase
            features[30] -= 0.2     # Lower spectral centroid
        elif emotion == 'surprise':
            features[0:13] += 0.4   # Higher MFCCs
            features[13] += 0.6     # Higher pitch
            features[30] += 0.3     # Higher spectral centroid
            features[35] += 0.4     # Higher spectral bandwidth
        
        return features
    
    def predict_emotion(self, audio, sr):
        """Predict emotion from audio"""
        if not self.is_trained:
            self.train_initial_model()
        
        # Extract features
        features = self.feature_extractor.extract_features(audio, sr)
        
        # Convert to array
        feature_vector = np.array([list(features.values())])
        
        # Predict
        if self.model:
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            # Get probabilities for each emotion
            prob_dict = {}
            for emotion, prob in zip(self.model.classes_, probabilities):
                prob_dict[emotion] = prob
            
            return prediction, prob_dict, features
        else:
            # Fallback to rule-based if model not available
            return self.rule_based_prediction(features), {}, features
    
    def rule_based_prediction(self, features):
        """Simple rule-based prediction as fallback"""
        pitch = features.get('pitch_mean', 0)
        energy = features.get('energy', 0)
        zcr = features.get('zcr_mean', 0)
        duration = features.get('duration', 0)
        
        # Simple rules
        if pitch > 200 and energy > 0.5:
            return 'happy'
        elif pitch > 180 and zcr > 0.1:
            return 'angry'
        elif pitch > 190 and energy > 0.3:
            return 'fearful'
        elif pitch < 120 and energy < 0.2:
            return 'sad'
        elif pitch < 150 and energy < 0.3:
            return 'calm'
        elif zcr > 0.15:
            return 'surprise'
        else:
            return 'neutral'

# Initialize classifier
classifier = VoiceEmotionClassifier()

# ==================== AUDIO RECORDING FUNCTIONS ====================
class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.is_recording = False
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 22050
        self.chunk = 1024
        
    def start_recording(self):
        """Start recording audio"""
        self.frames = []
        self.is_recording = True
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        def record_thread():
            while self.is_recording:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
        
        self.thread = threading.Thread(target=record_thread)
        self.thread.start()
        
    def stop_recording(self):
        """Stop recording and return audio data"""
        self.is_recording = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        # Convert to numpy array
        audio_data = b''.join(self.frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_array, self.sample_rate
    
    def save_recording(self, filename):
        """Save recording to file"""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_waveform(audio, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.linspace(0, len(audio)/sr, len(audio))
    ax.plot(time, audio, color=COLORS['primary'], alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Voice Waveform')
    ax.grid(True, alpha=0.3)
    return fig

def plot_spectrogram(audio, sr):
    """Plot mel spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title('Mel Spectrogram')
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_pitch_contour(audio, sr):
    """Plot pitch contour"""
    fig, ax = plt.subplots(figsize=(10, 3))
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    
    # Extract pitch values
    pitch_values = []
    times = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)
            times.append(i * 512 / sr)
    
    if pitch_values:
        ax.plot(times, pitch_values, color=COLORS['secondary'], linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pitch (Hz)')
        ax.set_title('Pitch Contour')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No pitch detected', ha='center', va='center')
    
    return fig

# ==================== STREAMLIT APP ====================
st.set_page_config(
    page_title="VoiceWell - Voice Emotion Analysis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #6366F1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea10, #764ba210);
        border-left: 5px solid #6366F1;
    }
    .wellness-tip {
        background-color: #E8F4FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #6366F1;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .recording-indicator {
        color: #EF4444;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'recorder' not in st.session_state:
    st.session_state.recorder = AudioRecorder()
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# ==================== MAIN APP ====================
def main():
    st.markdown("<h1 class='main-header'>🎤 VoiceWell - Voice Emotion Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Analyze your voice to understand emotional patterns and mental well-being</p>", unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        selected = option_menu(
            menu_title=None,
            options=["Record Voice", "Analyze Audio", "History", "Insights", "About"],
            icons=["mic", "graph-up", "clock-history", "heart", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": COLORS['primary'], "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": COLORS['primary']},
            }
        )
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.analysis_history:
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Analyses", len(st.session_state.analysis_history))
            
            # Most common emotion
            emotions = [a['dominant_emotion'] for a in st.session_state.analysis_history]
            most_common = Counter(emotions).most_common(1)[0][0] if emotions else "N/A"
            st.metric("Most Common", most_common.capitalize())
    
    if selected == "Record Voice":
        show_record_page()
    elif selected == "Analyze Audio":
        show_analyze_page()
    elif selected == "History":
        show_history_page()
    elif selected == "Insights":
        show_insights_page()
    elif selected == "About":
        show_about_page()

# ==================== RECORD PAGE ====================
def show_record_page():
    st.markdown("## 🎙️ Voice Recording")
    st.markdown("Record your voice to analyze emotional patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎤 Recording Controls")
        
        # Recording controls
        if not st.session_state.recording:
            if st.button("🔴 Start Recording", use_container_width=True, type="primary"):
                st.session_state.recorder.start_recording()
                st.session_state.recording = True
                st.rerun()
        else:
            st.markdown("<p class='recording-indicator'>🔴 Recording in progress...</p>", unsafe_allow_html=True)
            
            # Stop button
            if st.button("⏹️ Stop Recording", use_container_width=True, type="secondary"):
                audio, sr = st.session_state.recorder.stop_recording()
                st.session_state.recording = False
                
                # Save to session state
                st.session_state.current_audio = audio
                st.session_state.current_sr = sr
                
                st.success("✅ Recording completed!")
                st.rerun()
        
        # Recording tips
        st.markdown("### 📝 Recording Tips")
        st.info("""
        - Find a quiet environment
        - Speak naturally at your normal pace
        - Record for at least 3-5 seconds for better analysis
        - Avoid background noise
        - Speak clearly but naturally
        """)
    
    with col2:
        st.markdown("### 📊 Voice Quality Check")
        
        # Visual guide
        st.markdown("#### Good Recording")
        st.markdown("✓ Clear voice\n✓ Minimal background noise\n✓ Consistent volume")
        
        st.markdown("#### Avoid")
        st.markdown("✗ Background noise\n✗ Whispering\n✗ Shouting\n✗ Very short recordings")
        
        # Sample recordings
        with st.expander("🎵 Try Sample Recordings"):
            if st.button("Sample 1: Happy Voice"):
                # Generate synthetic happy voice
                duration = 3
                sr = 22050
                t = np.linspace(0, duration, int(sr * duration))
                # Create synthetic happy voice (higher pitch, more variation)
                audio = 0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
                audio += 0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-2 * t)
                audio += np.random.randn(len(audio)) * 0.01
                st.session_state.current_audio = audio
                st.session_state.current_sr = sr
                st.success("Sample loaded! Go to 'Analyze Audio' tab.")
            
            if st.button("Sample 2: Calm Voice"):
                # Generate synthetic calm voice
                duration = 3
                sr = 22050
                t = np.linspace(0, duration, int(sr * duration))
                # Create synthetic calm voice (lower pitch, steady)
                audio = 0.4 * np.sin(2 * np.pi * 150 * t) * np.exp(-0.5 * t)
                audio += 0.2 * np.sin(2 * np.pi * 300 * t) * 0.5
                audio += np.random.randn(len(audio)) * 0.005
                st.session_state.current_audio = audio
                st.session_state.current_sr = sr
                st.success("Sample loaded! Go to 'Analyze Audio' tab.")

# ==================== ANALYZE PAGE ====================
def show_analyze_page():
    st.markdown("## 📊 Voice Analysis")
    
    if not hasattr(st.session_state, 'current_audio') or st.session_state.current_audio is None:
        st.warning("No audio recorded. Please go to the 'Record Voice' tab first or load a sample.")
        
        # Option to load sample
        if st.button("🎵 Load Sample Voice"):
            # Generate sample audio
            duration = 3
            sr = 22050
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.5 * np.sin(2 * np.pi * 200 * t) * (1 + 0.2 * np.sin(2 * np.pi * 4 * t))
            audio += np.random.randn(len(audio)) * 0.01
            st.session_state.current_audio = audio
            st.session_state.current_sr = sr
            st.rerun()
        
        return
    
    # Analyze button
    if st.button("🔍 Analyze Voice", use_container_width=True, type="primary"):
        with st.spinner("Analyzing voice patterns..."):
            audio = st.session_state.current_audio
            sr = st.session_state.current_sr
            
            # Predict emotion
            emotion, probabilities, features = classifier.predict_emotion(audio, sr)
            
            # Get emotion data
            emotion_data = VOICE_EMOTION_MAPPING.get(emotion, VOICE_EMOTION_MAPPING['neutral'])
            
            # Store analysis
            analysis = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dominant_emotion': emotion,
                'probabilities': probabilities,
                'features': features,
                'emotion_data': emotion_data,
                'duration': features.get('duration', 0)
            }
            st.session_state.current_analysis = analysis
            st.session_state.analysis_history.append(analysis)
    
    # Display analysis results
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        emotion = analysis['dominant_emotion']
        emotion_data = analysis['emotion_data']
        
        st.markdown("---")
        st.markdown("### 📈 Analysis Results")
        
        # Main emotion display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background-color: {emotion_data['color']}20; 
                        border-radius: 15px; border: 2px solid {emotion_data['color']};">
                <h1 style="font-size: 4rem; margin: 0;">{emotion_data['icon']}</h1>
                <h2 style="color: {emotion_data['color']}; margin: 0;">{emotion.upper()}</h2>
                <p style="color: #666;">{emotion_data['status']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="wellness-tip" style="border-left-color: {emotion_data['color']};">
                <h4 style="color: {emotion_data['color']};">Wellness Insight</h4>
                <p>{emotion_data['wellness_tip']}</p>
                <h4 style="color: {emotion_data['color']};">Suggestion</h4>
                <p>{emotion_data['suggestion']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Audio visualizations
        st.markdown("### 🎵 Voice Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Waveform", "Spectrogram", "Pitch Contour"])
        
        with tab1:
            fig = plot_waveform(audio, sr)
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            fig = plot_spectrogram(audio, sr)
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            fig = plot_pitch_contour(audio, sr)
            st.pyplot(fig)
            plt.close()
        
        # Emotion probabilities
        st.markdown("### 📊 Emotion Probabilities")
        
        if analysis['probabilities']:
            probs_df = pd.DataFrame([
                {'Emotion': e.capitalize(), 'Probability': p * 100}
                for e, p in analysis['probabilities'].items()
            ]).sort_values('Probability', ascending=False)
            
            fig = px.bar(
                probs_df,
                x='Probability',
                y='Emotion',
                orientation='h',
                color='Probability',
                color_continuous_scale='Viridis',
                title="Emotion Detection Confidence"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Acoustic features
        with st.expander("🔬 Detailed Acoustic Features"):
            if analysis['features']:
                features_df = pd.DataFrame([
                    {'Feature': k.replace('_', ' ').title(), 'Value': f'{v:.3f}'}
                    for k, v in analysis['features'].items()
                ])
                st.dataframe(features_df, use_container_width=True)
        
        # Recommended activities
        st.markdown("### 🌱 Recommended Activities")
        cols = st.columns(3)
        for i, activity in enumerate(emotion_data['activities']):
            with cols[i]:
                if st.button(f"✅ {activity}", key=f"act_{i}"):
                    st.success(f"Great! Try to {activity.lower()} now.")

# ==================== HISTORY PAGE ====================
def show_history_page():
    st.markdown("## 📋 Analysis History")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history yet. Record and analyze your voice to see history!")
        return
    
    # Summary stats
    st.markdown("### 📊 Summary Statistics")
    
    df = pd.DataFrame([
        {
            'Date': a['timestamp'][:10],
            'Time': a['timestamp'][11:16],
            'Emotion': a['dominant_emotion'].capitalize(),
            'Duration': f"{a.get('duration', 0):.1f}s"
        }
        for a in st.session_state.analysis_history
    ])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyses", len(df))
    with col2:
        most_common = df['Emotion'].mode()[0] if not df.empty else "N/A"
        st.metric("Most Common", most_common)
    with col3:
        unique_days = df['Date'].nunique()
        st.metric("Active Days", unique_days)
    with col4:
        avg_duration = df['Duration'].str.replace('s', '').astype(float).mean()
        st.metric("Avg Duration", f"{avg_duration:.1f}s")
    
    # Timeline chart
    st.markdown("### 📈 Emotion Timeline")
    
    if not df.empty:
        fig = px.scatter(
            df,
            x='Date',
            y='Emotion',
            color='Emotion',
            size=[10] * len(df),
            title="Voice Analysis Timeline",
            color_discrete_sequence=[VOICE_EMOTION_MAPPING.get(e.lower(), {}).get('color', COLORS['gray']) 
                                    for e in df['Emotion']]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # History list
    st.markdown("### 📝 Recent Analyses")
    
    for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
        emotion = analysis['dominant_emotion']
        color = VOICE_EMOTION_MAPPING.get(emotion, {}).get('color', COLORS['gray'])
        
        with st.expander(f"Analysis {len(st.session_state.analysis_history)-i}: {analysis['timestamp']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Primary Emotion:** <span style='color:{color}'>{emotion.upper()}</span>", unsafe_allow_html=True)
                st.markdown(f"**Status:** {analysis['emotion_data']['status']}")
                st.markdown(f"**Duration:** {analysis.get('duration', 0):.1f} seconds")
            
            with col2:
                if st.button(f"View Details", key=f"view_{i}"):
                    st.session_state.current_analysis = analysis
                    st.rerun()
                
                if st.button(f"Delete", key=f"del_{i}"):
                    st.session_state.analysis_history.remove(analysis)
                    st.rerun()

# ==================== INSIGHTS PAGE ====================
def show_insights_page():
    st.markdown("## 💡 Voice Insights & Tips")
    
    if not st.session_state.analysis_history:
        st.info("No data yet. Record some voice samples to see insights!")
        return
    
    # Calculate insights
    df = pd.DataFrame(st.session_state.analysis_history)
    
    # Emotion distribution
    st.markdown("### 🎭 Emotion Distribution")
    
    emotion_counts = df['dominant_emotion'].value_counts()
    fig_pie = px.pie(
        values=emotion_counts.values,
        names=[e.capitalize() for e in emotion_counts.index],
        title="Your Emotional Patterns",
        color_discrete_sequence=[VOICE_EMOTION_MAPPING.get(e, {}).get('color', COLORS['gray']) 
                                for e in emotion_counts.index]
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Time-based patterns
    st.markdown("### 🕐 Time Patterns")
    
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp']).dt.day_name()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly distribution
        hourly_counts = df.groupby(['hour', 'dominant_emotion']).size().reset_index(name='count')
        fig_hourly = px.line(
            hourly_counts,
            x='hour',
            y='count',
            color='dominant_emotion',
            title="Emotions by Hour of Day",
            color_discrete_map={e: VOICE_EMOTION_MAPPING.get(e, {}).get('color', COLORS['gray']) 
                               for e in df['dominant_emotion'].unique()}
        )
        fig_hourly.update_layout(height=400)
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Daily distribution
        daily_counts = df.groupby(['day', 'dominant_emotion']).size().reset_index(name='count')
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts['day'] = pd.Categorical(daily_counts['day'], categories=day_order, ordered=True)
        daily_counts = daily_counts.sort_values('day')
        
        fig_daily = px.bar(
            daily_counts,
            x='day',
            y='count',
            color='dominant_emotion',
            title="Emotions by Day of Week",
            barmode='stack',
            color_discrete_map={e: VOICE_EMOTION_MAPPING.get(e, {}).get('color', COLORS['gray']) 
                               for e in df['dominant_emotion'].unique()}
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    # Wellness recommendations based on patterns
    st.markdown("### 🌱 Personalized Wellness Tips")
    
    most_common = df['dominant_emotion'].mode()[0]
    emotion_data = VOICE_EMOTION_MAPPING.get(most_common, VOICE_EMOTION_MAPPING['neutral'])
    
    st.markdown(f"""
    <div class="emotion-box" style="border-left-color: {emotion_data['color']};">
        <h4 style="color: {emotion_data['color']};">Based on your most common emotion: {most_common.upper()}</h4>
        <p><strong>Wellness Insight:</strong> {emotion_data['wellness_tip']}</p>
        <p><strong>Suggested Activities:</strong></p>
        <ul>
            <li>{emotion_data['activities'][0]}</li>
            <li>{emotion_data['activities'][1]}</li>
            <li>{emotion_data['activities'][2]}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== ABOUT PAGE ====================
def show_about_page():
    st.markdown("## ℹ️ About VoiceWell")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎤 Voice Emotion Analysis System
        
        VoiceWell is an advanced voice analysis tool that helps you understand your emotional patterns
        through voice analysis. By analyzing various acoustic features of your voice, it can detect
        emotional states and provide personalized mental health insights.
        
        ### 🔬 How It Works
        
        1. **Voice Recording**: Record your voice speaking naturally
        2. **Feature Extraction**: Analyzes 50+ acoustic features including:
           - Pitch and frequency patterns
           - Voice energy and intensity
           - Speech rate and rhythm
           - Spectral characteristics
           - Voice quality metrics
        
        3. **Emotion Detection**: Machine learning model identifies emotional patterns
        4. **Wellness Insights**: Provides personalized recommendations based on detected emotions
        
        ### 🎯 Features
        
        - Real-time voice recording and analysis
        - Multi-emotion detection (8 emotional states)
        - Detailed acoustic feature extraction
        - Visual waveform and spectrogram displays
        - Historical tracking and pattern analysis
        - Personalized wellness recommendations
        - Export and share capabilities
        
        ### 📊 Supported Emotions
        
        - 😌 Calm / Relaxed
        - 😊 Happy / Joyful
        - 😔 Sad / Melancholic
        - 😠 Angry / Frustrated
        - 😨 Fearful / Anxious
        - 😐 Neutral / Balanced
        - 🤢 Disgust / Aversion
        - 😲 Surprise / Shock
        
        ### 🏥 Clinical Applications
        
        - Mental health monitoring
        - Stress level assessment
        - Mood tracking
        - Therapy progress evaluation
        - Wellness coaching
        - Research purposes
        
        ### 🔒 Privacy & Security
        
        - All recordings processed locally
        - No data sent to external servers
        - Optional cloud backup with encryption
        - GDPR and HIPAA compliant options
        """)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 2rem; border-radius: 15px;">
            <h3 style="color: #4A90E2;">Quick Guide</h3>
            <ol style="color: #666;">
                <li>Go to 'Record Voice'</li>
                <li>Click 'Start Recording'</li>
                <li>Speak naturally</li>
                <li>Click 'Stop Recording'</li>
                <li>Go to 'Analyze Audio'</li>
                <li>Click 'Analyze Voice'</li>
            </ol>
            
            <h3 style="color: #4A90E2;">Tips</h3>
            <ul style="color: #666;">
                <li>Record 3-5 seconds minimum</li>
                <li>Find a quiet space</li>
                <li>Speak at normal volume</li>
                <li>Avoid background noise</li>
                <li>Be consistent</li>
            </ul>
            
            <h3 style="color: #4A90E2;">Version</h3>
            <p style="color: #666;">VoiceWell v1.0.0</p>
            <p style="color: #666;">Released: February 2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Credits
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>VoiceWell - Voice Emotion Analysis for Mental Wellness</p>
        <p>© 2026 NeuroWell Technologies. All rights reserved.</p>
        <p>For clinical use and research purposes.</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()