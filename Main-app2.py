import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import hashlib
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import base64
from pathlib import Path
import calendar
from collections import Counter
from fpdf import FPDF
import tempfile
import matplotlib.pyplot as plt
import io
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import librosa
import soundfile as sf
from scipy import stats
import wave
import struct
import audioop
import time
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

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
    'hospital_warm': '#FFB81C',
    'text_primary': '#6366F1',
    'text_secondary': '#10B981',
    'text_accent': '#F59E0B'
}

EMOTION_COLORS = {
    'angry': '#EF4444',
    'disgust': '#10B981',
    'fear': '#8B5CF6',
    'happy': '#F59E0B',
    'neutral': '#6B7280',
    'sad': '#3B82F6',
    'surprise': '#EC4899'
}

# ==================== ENHANCED VOICE ANALYSIS CONFIGURATION ====================
VOICE_EMOTION_MAPPING = {
    'calm': {
        'status': '😌 Calm / Relaxed',
        'suggestion': 'Your voice indicates a calm state. This is excellent for mindfulness and clear thinking.',
        'activities': ['Practice meditation', 'Engage in deep breathing', 'Listen to soothing music'],
        'color': '#4A90E2',
        'wellness_tip': 'Calmness is a superpower. Use it to make thoughtful decisions.',
        'icon': '😌',
        'category': 'Positive',
        'clinical_term': 'Relaxed State',
        'intervention': 'Mindfulness-based stress reduction'
    },
    'happy': {
        'status': '😊 Happy / Energetic',
        'suggestion': 'Your voice conveys happiness! Channel this positive energy into creative activities.',
        'activities': ['Share your joy with others', 'Start a creative project', 'Exercise'],
        'color': '#4CAF50',
        'wellness_tip': 'Happiness is contagious. Spread it to those around you.',
        'icon': '😊',
        'category': 'Positive',
        'clinical_term': 'Elevated Mood',
        'intervention': 'Positive psychology interventions'
    },
    'sad': {
        'status': '😔 Sad / Low Energy',
        'suggestion': 'Your voice suggests low mood. Be gentle with yourself today.',
        'activities': ['Reach out to a friend', 'Practice self-care', 'Gentle stretching'],
        'color': '#4169E1',
        'wellness_tip': 'It\'s okay to not be okay. Give yourself permission to feel.',
        'icon': '😔',
        'category': 'Negative',
        'clinical_term': 'Depressed Mood',
        'intervention': 'Behavioral activation therapy'
    },
    'angry': {
        'status': '😠 Angry / Agitated',
        'suggestion': 'Your voice shows signs of agitation. Take a moment to breathe.',
        'activities': ['Count to ten', 'Go for a walk', 'Write down your feelings'],
        'color': '#EF4444',
        'wellness_tip': 'Anger is a signal, not a solution. What is it telling you?',
        'icon': '😠',
        'category': 'Negative',
        'clinical_term': 'Irritability',
        'intervention': 'Anger management techniques'
    },
    'fearful': {
        'status': '😨 Fearful / Anxious',
        'suggestion': 'Your voice indicates anxiety. Practice grounding techniques.',
        'activities': ['Box breathing', 'Grounding exercises', 'Talk to someone'],
        'color': '#8B5CF6',
        'wellness_tip': 'Fear is future-focused. Bring yourself back to the present.',
        'icon': '😨',
        'category': 'Negative',
        'clinical_term': 'Anxiety Response',
        'intervention': 'Anxiety management and grounding'
    },
    'neutral': {
        'status': '😐 Neutral / Balanced',
        'suggestion': 'Your voice sounds balanced. This is perfect for focused work.',
        'activities': ['Start a task', 'Read something new', 'Practice mindfulness'],
        'color': '#6B7280',
        'wellness_tip': 'Balance is the key to sustainable well-being.',
        'icon': '😐',
        'category': 'Neutral',
        'clinical_term': 'Baseline State',
        'intervention': 'Maintain current coping strategies'
    },
    'surprised': {
        'status': '😲 Surprised / Alert',
        'suggestion': 'Your voice shows surprise or alertness. Process what you\'re experiencing.',
        'activities': ['Pause and reflect', 'Journal about it', 'Share your experience'],
        'color': '#F59E0B',
        'wellness_tip': 'Surprise opens the door to curiosity and learning.',
        'icon': '😲',
        'category': 'Neutral',
        'clinical_term': 'Startle Response',
        'intervention': 'Mindful observation'
    }
}

# ==================== ENHANCED TEXT ANALYSIS CONFIGURATION ====================
TEXT_EMOTION_MAPPING = {
    'joy': {
        'status': '😊 Joyful / Positive',
        'suggestion': 'Your text conveys positivity! This is great for mental well-being.',
        'activities': ['Express gratitude', 'Share your positivity', 'Engage in creative writing'],
        'color': '#4CAF50',
        'wellness_tip': 'Positive self-talk reinforces emotional resilience.',
        'icon': '😊',
        'category': 'Positive'
    },
    'sadness': {
        'status': '😔 Sadness / Melancholy',
        'suggestion': 'Your text shows signs of sadness. Consider reaching out to someone you trust.',
        'activities': ['Self-care routine', 'Connect with a loved one', 'Journal your feelings'],
        'color': '#4169E1',
        'wellness_tip': 'It\'s okay to feel sad. Acknowledging it is the first step to healing.',
        'icon': '😔',
        'category': 'Negative'
    },
    'anger': {
        'status': '😤 Anger / Frustration',
        'suggestion': 'Notice the anger in your text. Take a moment to breathe before responding.',
        'activities': ['Deep breathing', 'Physical exercise', 'Count to ten'],
        'color': '#EF4444',
        'wellness_tip': 'Anger often signals unmet needs. What might you need right now?',
        'icon': '😠',
        'category': 'Negative'
    },
    'fear': {
        'status': '😨 Fear / Anxiety',
        'suggestion': 'Your text suggests anxiety. Practice grounding techniques.',
        'activities': ['Box breathing', 'Progressive muscle relaxation', 'Grounding exercises'],
        'color': '#800080',
        'wellness_tip': 'Fear is future-focused. Bring yourself back to the present moment.',
        'icon': '😰',
        'category': 'Negative'
    },
    'surprise': {
        'status': '😲 Surprise / Shock',
        'suggestion': 'Your text indicates surprise. Take time to process new information.',
        'activities': ['Pause and reflect', 'Journal about it', 'Discuss with someone'],
        'color': '#FFA500',
        'wellness_tip': 'Surprise can be an invitation to curiosity.',
        'icon': '😲',
        'category': 'Neutral'
    },
    'trust': {
        'status': '🤝 Trust / Confidence',
        'suggestion': 'Your text shows trust and confidence. This builds strong relationships.',
        'activities': ['Build on this trust', 'Help others', 'Share your confidence'],
        'color': '#3B82F6',
        'wellness_tip': 'Trust is the foundation of emotional well-being.',
        'icon': '🤝',
        'category': 'Positive'
    },
    'anticipation': {
        'status': '🔮 Anticipation / Hope',
        'suggestion': 'You\'re looking forward to something. Channel this positive energy.',
        'activities': ['Plan ahead', 'Set goals', 'Visualize success'],
        'color': '#F59E0B',
        'wellness_tip': 'Anticipation can be a source of motivation and hope.',
        'icon': '🔮',
        'category': 'Positive'
    }
}

# ==================== MENTAL HEALTH INSIGHTS ====================
MENTAL_HEALTH_MAPPING = {
    'angry': {
        'status': '😤 High Stress / Irritability',
        'suggestion': 'Consider deep breathing exercises or stepping away for a moment. Try the 5-4-3-2-1 grounding technique.',
        'activities': ['Take 5 deep breaths', 'Go for a short walk', 'Listen to calming music'],
        'color': '#FF4B4B',
        'wellness_tip': 'Anger often signals unmet needs or boundaries. What might you need right now?',
        'icon': '😠',
        'category': 'Negative',
        'clinical_term': 'Elevated Irritability',
        'intervention': 'Cognitive Behavioral Therapy techniques for anger management'
    },
    'disgust': {
        'status': '🤢 Discomfort / Aversion',
        'suggestion': 'This might indicate something in your environment is bothering you. Identify and address the source.',
        'activities': ['Fresh air break', 'Clean/organize your space', 'Mindful observation'],
        'color': '#9ACD32',
        'wellness_tip': 'Disgust can be a protective emotion. Honor what your body is telling you.',
        'icon': '🤢',
        'category': 'Negative',
        'clinical_term': 'Aversive Response',
        'intervention': 'Exposure therapy and sensory integration techniques'
    },
    'fear': {
        'status': '😰 Anxiety / Apprehension',
        'suggestion': 'Practice grounding techniques. Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.',
        'activities': ['Box breathing (4-4-4-4)', 'Progressive muscle relaxation', 'Talk to someone you trust'],
        'color': '#800080',
        'wellness_tip': 'Fear is future-focused. Bring yourself back to the present moment.',
        'icon': '😨',
        'category': 'Negative',
        'clinical_term': 'Anxiety Response',
        'intervention': 'Anxiety management techniques and grounding exercises'
    },
    'happy': {
        'status': '😊 Positive / Content',
        'suggestion': 'Great! This is a good state for productivity and connection. Share your positivity with others.',
        'activities': ['Express gratitude', 'Help someone else', 'Engage in a creative activity'],
        'color': '#4CAF50',
        'wellness_tip': 'Savor this moment. What contributed to your happiness today?',
        'icon': '😊',
        'category': 'Positive',
        'clinical_term': 'Euthymic State',
        'intervention': 'Positive psychology interventions and gratitude practices'
    },
    'neutral': {
        'status': '😐 Calm / Balanced',
        'suggestion': 'You\'re in a balanced state. This is perfect for mindfulness or starting a new task.',
        'activities': ['Start that project you\'ve been putting off', 'Practice mindfulness', 'Read or learn something new'],
        'color': '#808080',
        'wellness_tip': 'Neutral doesn\'t mean empty. It\'s a peaceful baseline for well-being.',
        'icon': '😐',
        'category': 'Neutral',
        'clinical_term': 'Baseline Affective State',
        'intervention': 'Maintenance of current coping strategies'
    },
    'sad': {
        'status': '😔 Low Mood / Melancholy',
        'suggestion': 'Be gentle with yourself. This is a valid emotion that deserves acknowledgment and self-compassion.',
        'activities': ['Self-care routine', 'Connect with a loved one', 'Gentle exercise or stretching'],
        'color': '#4169E1',
        'wellness_tip': 'Sadness slows us down to process and heal. What do you need right now?',
        'icon': '😔',
        'category': 'Negative',
        'clinical_term': 'Depressed Mood',
        'intervention': 'Behavioral activation and social connection'
    },
    'surprise': {
        'status': '😲 Alert / Stimulated',
        'suggestion': 'Your nervous system is activated. Take a moment to assess if this is positive or needs attention.',
        'activities': ['Pause and breathe', 'Journal about what surprised you', 'Channel the energy creatively'],
        'color': '#FFA500',
        'wellness_tip': 'Surprise can be an invitation to curiosity. What can you learn from this moment?',
        'icon': '😲',
        'category': 'Neutral',
        'clinical_term': 'Startle Response',
        'intervention': 'Mindful observation and cognitive reframing'
    }
}

# ==================== ENHANCED VOICE ANALYZER CLASS ====================
class EnhancedVoiceEmotionAnalyzer:
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 20  # Increased MFCC features for better accuracy
        self.frame_length = 2048
        self.hop_length = 512
        
        # Advanced emotion thresholds with more nuanced parameters
        self.emotion_thresholds = {
            'happy': {
                'pitch_mean': (180, 320),
                'pitch_std': (40, 100),
                'energy_mean': (0.08, 0.25),
                'energy_std': (0.03, 0.15),
                'speech_rate': (140, 220),
                'spectral_centroid': (2000, 3500),
                'zcr_mean': (0.05, 0.15)
            },
            'sad': {
                'pitch_mean': (80, 160),
                'pitch_std': (15, 40),
                'energy_mean': (0.01, 0.08),
                'energy_std': (0.005, 0.03),
                'speech_rate': (70, 120),
                'spectral_centroid': (800, 1800),
                'zcr_mean': (0.02, 0.08)
            },
            'angry': {
                'pitch_mean': (200, 400),
                'pitch_std': (60, 150),
                'energy_mean': (0.15, 0.4),
                'energy_std': (0.08, 0.25),
                'speech_rate': (160, 250),
                'spectral_centroid': (2500, 4500),
                'zcr_mean': (0.1, 0.25)
            },
            'fearful': {
                'pitch_mean': (180, 350),
                'pitch_std': (50, 120),
                'energy_mean': (0.03, 0.15),
                'energy_std': (0.02, 0.1),
                'speech_rate': (120, 190),
                'spectral_centroid': (1800, 3000),
                'zcr_mean': (0.06, 0.18)
            },
            'calm': {
                'pitch_mean': (100, 180),
                'pitch_std': (20, 50),
                'energy_mean': (0.01, 0.06),
                'energy_std': (0.005, 0.02),
                'speech_rate': (80, 130),
                'spectral_centroid': (1000, 2000),
                'zcr_mean': (0.02, 0.06)
            },
            'surprised': {
                'pitch_mean': (250, 450),
                'pitch_std': (80, 180),
                'energy_mean': (0.1, 0.3),
                'energy_std': (0.05, 0.2),
                'speech_rate': (150, 230),
                'spectral_centroid': (2800, 5000),
                'zcr_mean': (0.08, 0.22)
            },
            'neutral': {
                'pitch_mean': (120, 200),
                'pitch_std': (25, 60),
                'energy_mean': (0.02, 0.1),
                'energy_std': (0.01, 0.05),
                'speech_rate': (100, 150),
                'spectral_centroid': (1200, 2200),
                'zcr_mean': (0.03, 0.1)
            }
        }
        
        # Feature weights for different emotions
        self.feature_weights = {
            'pitch_mean': 0.15,
            'pitch_std': 0.10,
            'energy_mean': 0.12,
            'energy_std': 0.08,
            'speech_rate': 0.10,
            'spectral_centroid': 0.08,
            'zcr_mean': 0.07,
            'mfcc': 0.30  # Combined weight for all MFCCs
        }
        
    def extract_enhanced_features(self, audio_path):
        """Extract comprehensive acoustic features with error handling"""
        try:
            # Load audio with duration limit
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30)
            
            # Check for valid audio
            if len(y) < self.sample_rate:  # Less than 1 second
                return None, None, None
            
            features = {}
            
            # 1. Pitch features with better estimation
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            
            # Get only pitches with significant magnitude
            pitch_mask = magnitudes > np.median(magnitudes)
            pitch_values = pitches[pitch_mask]
            pitch_values = pitch_values[pitch_values > 0]
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_max'] = float(np.max(pitch_values))
                features['pitch_min'] = float(np.min(pitch_values))
                features['pitch_range'] = float(features['pitch_max'] - features['pitch_min'])
                features['pitch_variability'] = float(features['pitch_std'] / features['pitch_mean'] if features['pitch_mean'] > 0 else 0)
            else:
                features['pitch_mean'] = 150.0
                features['pitch_std'] = 50.0
                features['pitch_max'] = 200.0
                features['pitch_min'] = 100.0
                features['pitch_range'] = 100.0
                features['pitch_variability'] = 0.33
            
            # 2. Energy features with multiple statistics
            rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            features['energy_max'] = float(np.max(rms))
            features['energy_min'] = float(np.min(rms))
            features['energy_range'] = float(features['energy_max'] - features['energy_min'])
            features['energy_variability'] = float(features['energy_std'] / features['energy_mean'] if features['energy_mean'] > 0 else 0)
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
            
            # 4. MFCC features (increased to 20)
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.frame_length, hop_length=self.hop_length
            )
            
            for i in range(self.n_mfcc):
                features[f'mfcc_mean_{i+1}'] = float(np.mean(mfccs[i]))
                features[f'mfcc_std_{i+1}'] = float(np.std(mfccs[i]))
                features[f'mfcc_max_{i+1}'] = float(np.max(mfccs[i]))
                features[f'mfcc_min_{i+1}'] = float(np.min(mfccs[i]))
            
            # 5. Voice quality features
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 6. Harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(y)
            features['harmonic_ratio'] = float(np.sum(np.abs(harmonic)) / (np.sum(np.abs(percussive)) + 1e-10))
            
            # 7. Speech rate and rhythm
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            features['tempo'] = float(tempo)
            
            # Syllable rate estimation
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if len(onset_frames) > 1:
                duration = len(y) / sr
                speech_rate = len(onset_frames) / duration * 60
                features['speech_rate'] = float(speech_rate)
            else:
                features['speech_rate'] = 120.0
            
            # 8. Voice breaks and pauses
            non_silent = librosa.effects.split(y, top_db=30)
            if len(non_silent) > 0:
                total_speech = np.sum([end - start for start, end in non_silent])
                silence_duration = len(y) - total_speech
                features['speech_percentage'] = float(total_speech / len(y) * 100)
                features['pause_count'] = len(non_silent) - 1
                features['avg_pause_duration'] = float(silence_duration / max(features['pause_count'], 1))
            else:
                features['speech_percentage'] = 100.0
                features['pause_count'] = 0
                features['avg_pause_duration'] = 0.0
            
            return features, y, sr
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None, None, None
    
    def detect_emotion_ml(self, features):
        """Detect emotion using machine learning approach with weighted features"""
        if not features:
            return 'neutral', 0.5, {emotion: 0.5/len(VOICE_EMOTION_MAPPING) for emotion in VOICE_EMOTION_MAPPING.keys()}
        
        emotion_scores = {emotion: 0.0 for emotion in VOICE_EMOTION_MAPPING.keys()}
        total_weight = 0.0
        
        # Calculate scores for each feature
        for emotion, thresholds in self.emotion_thresholds.items():
            score = 0.0
            weight_sum = 0.0
            
            # Check each feature
            for feature, (low, high) in thresholds.items():
                if feature in features:
                    value = features[feature]
                    
                    # Calculate how well the value fits in the range
                    if low <= value <= high:
                        # Perfect match
                        feature_score = 1.0
                    elif value < low:
                        # Below range - distance-based score
                        feature_score = max(0, 1 - (low - value) / low)
                    else:
                        # Above range - distance-based score
                        feature_score = max(0, 1 - (value - high) / high)
                    
                    # Apply feature weight
                    weight = self.feature_weights.get(feature, 0.05)
                    score += feature_score * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                emotion_scores[emotion] = score / weight_sum
            else:
                emotion_scores[emotion] = 0.5
        
        # Add MFCC-based scoring
        mfcc_features = [features.get(f'mfcc_mean_{i}', 0) for i in range(1, 13)]
        
        # Happy MFCC pattern
        if np.mean(mfcc_features[:4]) > -50 and np.std(mfcc_features[:4]) > 30:
            emotion_scores['happy'] += 0.15
        # Sad MFCC pattern
        elif np.mean(mfcc_features[:4]) < -100:
            emotion_scores['sad'] += 0.15
        # Angry MFCC pattern
        elif np.mean(mfcc_features[4:8]) > 50:
            emotion_scores['angry'] += 0.15
        # Fearful MFCC pattern
        elif np.std(mfcc_features) > 60:
            emotion_scores['fearful'] += 0.15
        # Calm MFCC pattern
        elif np.std(mfcc_features) < 30:
            emotion_scores['calm'] += 0.15
        else:
            emotion_scores['neutral'] += 0.15
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total
        
        # Apply softmax for better probability distribution
        exp_scores = np.exp([emotion_scores[e] for e in emotion_scores])
        exp_scores = exp_scores / np.sum(exp_scores)
        
        for i, emotion in enumerate(emotion_scores):
            emotion_scores[emotion] = float(exp_scores[i])
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        
        return dominant_emotion, confidence, emotion_scores
    
    def analyze_voice_health_advanced(self, features):
        """Advanced voice health analysis with clinical insights"""
        health_indicators = []
        
        if not features:
            return health_indicators
        
        # Check for vocal strain
        if features.get('pitch_mean', 0) > 300 and features.get('energy_mean', 0) > 0.15:
            if features.get('zcr_mean', 0) > 0.1:
                health_indicators.append({
                    'issue': 'Significant vocal strain detected',
                    'suggestion': 'Rest your voice completely for 24 hours. Avoid whispering (it strains more!). Stay hydrated and consider steam inhalation.',
                    'severity': 'high',
                    'clinical_term': 'Vocal Hyperfunction',
                    'exercises': ['Semi-occluded vocal tract exercises', 'Lip trills', 'Humming']
                })
            else:
                health_indicators.append({
                    'issue': 'Moderate vocal strain',
                    'suggestion': 'Take frequent voice breaks. Drink warm water with honey. Avoid clearing throat aggressively.',
                    'severity': 'moderate',
                    'clinical_term': 'Mild Vocal Fatigue',
                    'exercises': ['Gentle humming', 'Straw phonation', 'Deep breathing']
                })
        
        # Check for breathiness (possible vocal cord issues)
        if features.get('energy_mean', 0) < 0.03 and features.get('zcr_mean', 0) > 0.12:
            health_indicators.append({
                'issue': 'Breathy voice quality',
                'suggestion': 'This may indicate incomplete vocal fold closure. Practice sustained vowel sounds and consult an ENT if persistent.',
                'severity': 'moderate',
                'clinical_term': 'Breathy Dysphonia',
                'exercises': ['Sustained "ah" vowel', 'Resonant voice therapy', 'Semi-occluded exercises']
            })
        
        # Check for monotone (possible depression indicator)
        if features.get('pitch_std', 100) < 25:
            health_indicators.append({
                'issue': 'Reduced pitch variability',
                'suggestion': 'This can indicate emotional flatness. Try reading aloud with exaggerated expression or singing exercises.',
                'severity': 'moderate',
                'clinical_term': 'Monopitch',
                'exercises': ['Pitch glides', 'Reading with expression', 'Singing exercises']
            })
        
        # Check for tremor
        pitch_variability = features.get('pitch_variability', 0.3)
        if pitch_variability > 0.6 and features.get('pitch_std', 0) > 70:
            health_indicators.append({
                'issue': 'Vocal tremor detected',
                'suggestion': 'Could indicate stress, fatigue, or neurological factors. Practice diaphragmatic breathing and reduce caffeine.',
                'severity': 'high' if pitch_variability > 0.8 else 'moderate',
                'clinical_term': 'Vocal Tremor',
                'exercises': ['Diaphragmatic breathing', 'Sustained vowels', 'Relaxation techniques']
            })
        
        # Check for harshness
        if features.get('spectral_centroid_mean', 2000) > 3500 and features.get('zcr_mean', 0) > 0.15:
            health_indicators.append({
                'issue': 'Harsh voice quality',
                'suggestion': 'May indicate vocal abuse or reflux. Avoid shouting, and try semi-occluded vocal tract exercises.',
                'severity': 'moderate',
                'clinical_term': 'Vocal Harshness',
                'exercises': ['Easy onset phonation', 'Semi-occluded exercises', 'Hydration']
            })
        
        return health_indicators

# ==================== ENHANCED TEXT ANALYZER CLASS ====================
class EnhancedTextEmotionAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_patterns = self.load_enhanced_emotion_patterns()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
    def load_enhanced_emotion_patterns(self):
        """Load comprehensive emotion patterns with weights and context"""
        return {
            'joy': {
                'words': {
                    'happy': 0.8, 'joy': 1.0, 'delighted': 0.9, 'pleased': 0.7,
                    'grateful': 0.8, 'thankful': 0.8, 'hopeful': 0.6, 'optimistic': 0.7,
                    'positive': 0.6, 'good': 0.5, 'great': 0.7, 'excellent': 0.8,
                    'wonderful': 0.9, 'fantastic': 0.9, 'amazing': 0.8, 'beautiful': 0.6,
                    'love': 0.9, 'loving': 0.8, 'glad': 0.7, 'cheerful': 0.8,
                    'excited': 0.8, 'thrilled': 0.9, 'blessed': 0.7, 'celebrate': 0.8,
                    'appreciate': 0.7, 'enjoy': 0.6, 'smile': 0.6, 'laugh': 0.7,
                    'bliss': 1.0, 'ecstatic': 1.0, 'elated': 0.9, 'content': 0.6
                },
                'phrases': {
                    'i am happy': 0.9, 'feeling good': 0.8, 'so happy': 0.9,
                    'very happy': 0.9, 'made my day': 1.0, 'best day': 0.9,
                    'couldn\'t be better': 0.9, 'on top of the world': 1.0,
                    'over the moon': 1.0, 'in heaven': 0.9, 'perfect day': 0.8
                },
                'intensifiers': ['so', 'very', 'extremely', 'really', 'absolutely', 'totally']
            },
            'sadness': {
                'words': {
                    'sad': 0.8, 'depressed': 1.0, 'unhappy': 0.7, 'miserable': 0.9,
                    'gloomy': 0.6, 'hopeless': 1.0, 'devastated': 1.0, 'heartbroken': 1.0,
                    'grief': 1.0, 'mourning': 0.9, 'distressed': 0.8, 'tearful': 0.7,
                    'crying': 0.7, 'sorrow': 0.9, 'despair': 1.0, 'lonely': 0.8,
                    'alone': 0.6, 'hurt': 0.7, 'pain': 0.7, 'suffering': 0.8,
                    'blue': 0.5, 'down': 0.6, 'empty': 0.8, 'numb': 0.7,
                    'grieving': 0.9, 'melancholy': 0.8, 'somber': 0.7
                },
                'phrases': {
                    'feel sad': 0.8, 'so sad': 0.8, 'very sad': 0.8,
                    'i am sad': 0.7, 'want to cry': 0.8, 'no hope': 0.9,
                    'give up': 0.9, 'can\'t go on': 1.0, 'lost all hope': 1.0,
                    'broken heart': 0.9, 'in pain': 0.7
                }
            },
            'anger': {
                'words': {
                    'angry': 0.9, 'mad': 0.7, 'furious': 1.0, 'frustrated': 0.8,
                    'irritated': 0.7, 'annoyed': 0.6, 'hostile': 0.8, 'aggressive': 0.8,
                    'outraged': 1.0, 'resentful': 0.8, 'bitter': 0.7, 'frustration': 0.8,
                    'irritation': 0.7, 'hate': 0.9, 'hatred': 1.0, 'rage': 1.0,
                    'fury': 1.0, 'fuming': 0.9, 'seething': 0.9, 'livid': 0.9,
                    'enraged': 1.0, 'infuriated': 1.0, 'pissed': 0.8
                },
                'phrases': {
                    'i am angry': 0.8, 'so angry': 0.9, 'very angry': 0.9,
                    'makes me mad': 0.7, 'fed up': 0.7, 'had enough': 0.8,
                    'lose my mind': 0.8, 'drive me crazy': 0.7, 'pissed off': 0.8,
                    'sick of': 0.6, 'tired of': 0.5
                }
            },
            'fear': {
                'words': {
                    'afraid': 0.8, 'scared': 0.8, 'terrified': 1.0, 'anxious': 0.8,
                    'worried': 0.7, 'nervous': 0.7, 'panicked': 0.9, 'frightened': 0.8,
                    'alarmed': 0.7, 'dread': 0.9, 'fearful': 0.8, 'anxiety': 0.8,
                    'panic': 0.9, 'phobia': 0.8, 'stress': 0.6, 'stressed': 0.6,
                    'terrified': 1.0, 'horror': 0.8, 'paranoid': 0.7, 'uneasy': 0.6,
                    'apprehensive': 0.7, 'distressed': 0.7
                },
                'phrases': {
                    'i am afraid': 0.8, 'so scared': 0.9, 'very anxious': 0.8,
                    'worried about': 0.7, 'scared to death': 1.0, 'panic attack': 0.9,
                    'fear of': 0.7, 'can\'t breathe': 0.8, 'heart racing': 0.7
                }
            },
            'surprise': {
                'words': {
                    'surprised': 0.7, 'shocked': 0.8, 'amazed': 0.8, 'astonished': 0.8,
                    'stunned': 0.8, 'unexpected': 0.7, 'unanticipated': 0.7,
                    'sudden': 0.6, 'startled': 0.7, 'remarkable': 0.6,
                    'extraordinary': 0.6, 'wow': 0.7, 'oh': 0.5, 'ah': 0.5,
                    'incredible': 0.7, 'unbelievable': 0.8, 'astounded': 0.8
                },
                'phrases': {
                    'i am surprised': 0.7, 'so surprised': 0.8,
                    'can\'t believe': 0.8, 'no way': 0.7, 'oh my god': 0.7,
                    'holy cow': 0.7, 'oh my': 0.6, 'you\'re kidding': 0.7
                }
            },
            'trust': {
                'words': {
                    'trust': 0.8, 'confident': 0.7, 'believe': 0.6, 'faith': 0.7,
                    'reliable': 0.7, 'dependable': 0.7, 'assured': 0.6,
                    'certain': 0.6, 'sure': 0.5, 'confidence': 0.7,
                    'trusting': 0.7, 'trustworthy': 0.8, 'honest': 0.7,
                    'sincere': 0.6, 'loyal': 0.7, 'genuine': 0.6, 'authentic': 0.6
                },
                'phrases': {
                    'i trust': 0.8, 'i believe': 0.7, 'i am confident': 0.8,
                    'have faith': 0.7, 'count on': 0.6, 'depend on': 0.6,
                    'rely on': 0.6
                }
            },
            'anticipation': {
                'words': {
                    'expect': 0.6, 'anticipate': 0.7, 'await': 0.6,
                    'look forward': 0.8, 'prospective': 0.5, 'pending': 0.5,
                    'upcoming': 0.5, 'future': 0.4, 'expecting': 0.6,
                    'awaiting': 0.6, 'preparing': 0.5, 'hope': 0.7,
                    'hoping': 0.7, 'wish': 0.6, 'wishing': 0.6,
                    'excited for': 0.8, 'can\'t wait': 0.8, 'looking forward': 0.8
                },
                'phrases': {
                    'looking forward': 0.8, 'can\'t wait': 0.8,
                    'hope for': 0.7, 'excited about': 0.8, 'counting down': 0.7
                }
            }
        }
    
    def analyze_sentiment_enhanced(self, text):
        """Enhanced sentiment analysis combining multiple methods"""
        # VADER sentiment
        vader_scores = self.sia.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_scores = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Word-level sentiment (custom lexicon)
        words = text.lower().split()
        custom_pos = 0
        custom_neg = 0
        custom_neu = 0
        
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                              'beautiful', 'happy', 'joy', 'love', 'perfect', 'best', 'awesome',
                              'brilliant', 'outstanding', 'superb', 'terrific', 'splendid'])
        
        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
                              'sad', 'angry', 'upset', 'disappointed', 'frustrated', 'annoying',
                              'awful', 'dreadful', 'atrocious', 'abysmal', 'pathetic'])
        
        for word in words:
            word = word.strip('.,!?;:')
            if word in positive_words:
                custom_pos += 1
            elif word in negative_words:
                custom_neg += 1
            else:
                custom_neu += 1
        
        total_words = len(words) if words else 1
        custom_sentiment = (custom_pos - custom_neg) / total_words
        
        # Calculate confidence based on word matches
        confidence = min(1.0, (custom_pos + custom_neg) / max(total_words, 1) * 2)
        
        # Combine scores with dynamic weights based on confidence
        vader_weight = 0.4
        textblob_weight = 0.3
        custom_weight = 0.3
        
        combined_sentiment = {
            'compound': (vader_scores['compound'] * vader_weight + 
                        textblob_scores['polarity'] * textblob_weight + 
                        custom_sentiment * custom_weight),
            'positive': (vader_scores['pos'] * 0.4 + max(0, custom_sentiment) * 0.3 + 
                        (textblob_scores['polarity'] if textblob_scores['polarity'] > 0 else 0) * 0.3),
            'negative': (vader_scores['neg'] * 0.4 + max(0, -custom_sentiment) * 0.3 + 
                        (-textblob_scores['polarity'] if textblob_scores['polarity'] < 0 else 0) * 0.3),
            'neutral': vader_scores['neu'] * 0.4 + (1 - abs(custom_sentiment)) * 0.3 + 
                      (1 - abs(textblob_scores['polarity'])) * 0.3,
            'confidence': confidence
        }
        
        return combined_sentiment, vader_scores, textblob_scores
    
    def detect_emotions_weighted(self, text):
        """Detect emotions with weighted scoring and context"""
        text_lower = text.lower()
        words = nltk.word_tokenize(text_lower)
        sentences = nltk.sent_tokenize(text)
        
        emotion_scores = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            word_matches = 0
            phrase_matches = 0
            
            # Check individual words with weights
            for word, weight in patterns['words'].items():
                # Count occurrences with word boundaries
                count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
                if count > 0:
                    word_matches += count
                    # Base score from word matches
                    score += count * weight * 0.3
                    
                    # Bonus for repeated words (intensity)
                    if count > 1:
                        score += (count - 1) * weight * 0.1
            
            # Check phrases with weights
            for phrase, weight in patterns['phrases'].items():
                if phrase in text_lower:
                    phrase_matches += 1
                    score += weight * 0.5
            
            # Check for intensifiers
            if 'intensifiers' in patterns:
                for intensifier in patterns['intensifiers']:
                    for word in patterns['words']:
                        if f"{intensifier} {word}" in text_lower:
                            score += 0.3  # Higher bonus for intensified emotions
            
            # Check sentence-level patterns
            for sentence in sentences:
                sent_lower = sentence.lower()
                # Check if sentence is short and emotional (likely more intense)
                if len(sent_lower.split()) < 8:
                    for word in patterns['words']:
                        if word in sent_lower:
                            score += 0.2  # Extra weight for short emotional sentences
            
            # Normalize score based on text length
            word_count = len(words)
            if word_count > 0:
                score = score / (word_count ** 0.5)  # Square root normalization
            
            if score > 0:
                emotion_scores[emotion] = round(score, 3)
        
        # If no emotions detected, use sentiment analysis with fallback
        if not emotion_scores:
            combined_sentiment, vader, _ = self.analyze_sentiment_enhanced(text)
            
            # More nuanced emotion mapping based on sentiment
            compound = combined_sentiment['compound']
            
            if compound >= 0.5:
                emotion_scores['joy'] = 0.5
                emotion_scores['trust'] = 0.3
                emotion_scores['anticipation'] = 0.2
            elif compound <= -0.5:
                if 'angry' in text_lower or 'hate' in text_lower or 'mad' in text_lower:
                    emotion_scores['anger'] = 0.6
                    emotion_scores['sadness'] = 0.3
                elif 'scared' in text_lower or 'afraid' in text_lower or 'anxious' in text_lower:
                    emotion_scores['fear'] = 0.6
                    emotion_scores['sadness'] = 0.3
                else:
                    emotion_scores['sadness'] = 0.6
                    emotion_scores['fear'] = 0.3
            elif compound > 0.1:
                emotion_scores['trust'] = 0.4
                emotion_scores['anticipation'] = 0.3
                emotion_scores['joy'] = 0.2
            elif compound < -0.1:
                emotion_scores['fear'] = 0.4
                emotion_scores['sadness'] = 0.3
                emotion_scores['anger'] = 0.2
            else:
                emotion_scores['neutral'] = 0.7
                emotion_scores['trust'] = 0.2
        
        # Normalize scores to sum to 1
        total = sum(emotion_scores.values())
        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total
        
        return emotion_scores
    
    def extract_key_phrases_enhanced(self, text, num_phrases=5):
        """Enhanced key phrase extraction using multiple methods"""
        phrases = set()
        
        # Method 1: TF-IDF based with n-grams
        try:
            vectorizer = TfidfVectorizer(
                max_features=15, 
                stop_words='english', 
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.9
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases
            top_indices = scores.argsort()[-num_phrases*2:][::-1]
            for idx in top_indices:
                if scores[idx] > 0.1:
                    phrases.add(feature_names[idx])
        except Exception as e:
            print(f"TF-IDF error: {e}")
        
        # Method 2: POS-based noun phrases (more accurate)
        try:
            blob = TextBlob(text)
            noun_phrases = list(blob.noun_phrases)
            for phrase in noun_phrases[:3]:
                if len(phrase.split()) <= 3:  # Keep short phrases
                    phrases.add(phrase)
        except:
            pass
        
        # Method 3: Frequency-based with filtering
        words = text.lower().split()
        # Remove stopwords and punctuation
        filtered_words = [word.strip('.,!?;:') for word in words 
                         if word.strip('.,!?;:') not in self.stopwords 
                         and len(word.strip('.,!?;:')) > 2]
        
        word_counts = Counter(filtered_words)
        for word, count in word_counts.most_common(5):
            if count > 1:  # Only include if appears multiple times
                phrases.add(word)
        
        # Method 4: Emotion words (most impactful)
        for emotion, patterns in self.emotion_patterns.items():
            for word in patterns['words']:
                if word in text_lower and len(word) > 2:
                    phrases.add(word)
        
        # Convert to list and limit
        phrases_list = list(phrases)
        
        # Sort by length (prefer longer phrases first as they're more specific)
        phrases_list.sort(key=len, reverse=True)
        
        return phrases_list[:num_phrases]
    
    def assess_crisis_risk_enhanced(self, text):
        """Enhanced crisis risk assessment with severity scoring"""
        text_lower = text.lower()
        
        crisis_indicators = {
            'critical': [
                'suicide', 'kill myself', 'end my life', 'want to die',
                'self-harm', 'hurt myself', 'no reason to live', 'better off dead',
                'take my life', 'end it all', 'commit suicide', 'die tonight',
                'wish i was dead', 'want to disappear', 'can\'t do this anymore'
            ],
            'high': [
                'hopeless', 'can\'t go on', 'giving up', 'no way out',
                'tired of living', 'worthless', 'pointless', 'nothing matters',
                'no hope', 'can\'t take it', 'overwhelmed', 'drowning',
                'suffocating', 'falling apart', 'breaking down'
            ],
            'moderate': [
                'depressed', 'sad', 'lonely', 'alone', 'struggling',
                'anxious', 'scared', 'worried', 'stress', 'tired',
                'exhausted', 'burnout', 'overthinking', 'paranoid'
            ],
            'low': [
                'down', 'blue', 'unhappy', 'frustrated', 'upset',
                'disappointed', 'concerned', 'uneasy', 'meh', 'blah'
            ]
        }
        
        risk_level = 'none'
        indicators_found = []
        severity_score = 0
        
        for level, indicators in crisis_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    indicators_found.append(indicator)
                    
                    # Assign severity scores with weights
                    if level == 'critical':
                        severity_score += 10
                        risk_level = 'CRITICAL'
                    elif level == 'high':
                        severity_score += 5
                        if risk_level not in ['CRITICAL']:
                            risk_level = 'HIGH'
                    elif level == 'moderate':
                        severity_score += 2
                        if risk_level not in ['CRITICAL', 'HIGH']:
                            risk_level = 'MODERATE'
                    elif level == 'low':
                        severity_score += 1
                        if risk_level not in ['CRITICAL', 'HIGH', 'MODERATE']:
                            risk_level = 'LOW'
        
        # Check for multiple indicators (higher severity)
        unique_indicators = len(set(indicators_found))
        if unique_indicators >= 3:
            severity_score += unique_indicators
            
        return {
            'risk_level': risk_level,
            'indicators': list(set(indicators_found)),  # Remove duplicates
            'severity_score': severity_score,
            'requires_immediate_attention': risk_level in ['CRITICAL', 'HIGH'],
            'indicator_count': len(set(indicators_found))
        }
    
    def analyze_text_comprehensive(self, text):
        """Comprehensive text analysis combining all methods"""
        # Basic stats
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        word_count = len(words)
        char_count = len(text)
        sentence_count = len(sentences)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Advanced stats
        unique_words = len(set([w.lower() for w in words]))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Part of speech tagging for deeper analysis
        try:
            pos_tags = nltk.pos_tag(words)
            noun_count = len([tag for _, tag in pos_tags if tag.startswith('N')])
            verb_count = len([tag for _, tag in pos_tags if tag.startswith('V')])
            adj_count = len([tag for _, tag in pos_tags if tag.startswith('J')])
            adv_count = len([tag for _, tag in pos_tags if tag.startswith('R')])
            pronoun_count = len([tag for _, tag in pos_tags if tag.startswith('PRP')])
        except:
            noun_count = verb_count = adj_count = adv_count = pronoun_count = 0
        
        # Sentiment analysis (enhanced)
        combined_sentiment, vader_scores, textblob_scores = self.analyze_sentiment_enhanced(text)
        
        # Emotion detection (weighted)
        emotions_detected = self.detect_emotions_weighted(text)
        
        # Key phrases (enhanced)
        key_phrases = self.extract_key_phrases_enhanced(text)
        
        # Crisis risk (enhanced)
        crisis_risk = self.assess_crisis_risk_enhanced(text)
        
        # Determine dominant emotion with confidence
        if emotions_detected:
            dominant_emotion = max(emotions_detected, key=emotions_detected.get)
            dominant_score = emotions_detected[dominant_emotion]
            emotion_confidence = dominant_score
        else:
            # Fallback based on sentiment
            if combined_sentiment['compound'] >= 0.5:
                dominant_emotion = 'joy'
                emotion_confidence = 0.6
            elif combined_sentiment['compound'] <= -0.5:
                # Check for specific negative emotions
                text_lower = text.lower()
                if any(word in text_lower for word in ['angry', 'hate', 'mad', 'furious']):
                    dominant_emotion = 'anger'
                elif any(word in text_lower for word in ['scared', 'afraid', 'anxious', 'worried']):
                    dominant_emotion = 'fear'
                else:
                    dominant_emotion = 'sadness'
                emotion_confidence = 0.5
            elif combined_sentiment['compound'] > 0.1:
                dominant_emotion = 'trust'
                emotion_confidence = 0.4
            elif combined_sentiment['compound'] < -0.1:
                dominant_emotion = 'fear'
                emotion_confidence = 0.4
            else:
                dominant_emotion = 'neutral'
                emotion_confidence = 0.5
            dominant_score = emotion_confidence
        
        # Get wellness recommendations
        wellness = self.get_wellness_recommendations(dominant_emotion, text, combined_sentiment)
        
        return {
            'dominant_emotion': dominant_emotion,
            'dominant_score': dominant_score,
            'emotions_detected': emotions_detected,
            'sentiment': {
                'combined': combined_sentiment,
                'vader': vader_scores,
                'textblob': textblob_scores
            },
            'key_phrases': key_phrases,
            'crisis_risk': crisis_risk,
            'wellness': wellness,
            'statistics': {
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count,
                'avg_word_length': round(avg_word_length, 2),
                'avg_sentence_length': round(avg_sentence_length, 2),
                'lexical_diversity': round(lexical_diversity, 3),
                'unique_words': unique_words,
                'noun_count': noun_count,
                'verb_count': verb_count,
                'adj_count': adj_count,
                'adv_count': adv_count,
                'pronoun_count': pronoun_count
            },
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'full_text': text
        }
    
    def get_wellness_recommendations(self, dominant_emotion, text, sentiment_data):
        """Get personalized wellness recommendations based on multiple factors"""
        emotion_data = TEXT_EMOTION_MAPPING.get(dominant_emotion, {
            'status': '😐 Neutral',
            'suggestion': 'Your text appears neutral. This is a good baseline.',
            'activities': ['Practice mindfulness', 'Read something inspiring', 'Take a walk'],
            'color': '#808080',
            'wellness_tip': 'A balanced state is perfect for self-reflection.',
            'icon': '😐',
            'category': 'Neutral'
        })
        
        # Personalize based on sentiment intensity
        if sentiment_data and sentiment_data.get('compound', 0) < -0.5:
            emotion_data['suggestion'] += " Please consider reaching out to a mental health professional or trusted friend."
        elif sentiment_data and sentiment_data.get('compound', 0) < -0.3:
            emotion_data['suggestion'] += " Consider talking to someone you trust about your feelings."
        elif sentiment_data and sentiment_data.get('compound', 0) > 0.5:
            emotion_data['suggestion'] += " This positive energy is great for creative activities and connecting with others!"
        
        # Add context-specific suggestions
        if 'work' in text.lower() or 'job' in text.lower():
            emotion_data['activities'] = [
                'Take a short break', 
                'Practice desk stretches', 
                'Set boundaries for work-life balance'
            ]
        elif 'relationship' in text.lower() or 'partner' in text.lower() or 'friend' in text.lower():
            emotion_data['activities'] = [
                'Communicate openly', 
                'Practice active listening', 
                'Schedule quality time together'
            ]
        
        return emotion_data

# ==================== CUSTOM LOGO LOADER ====================
def get_base64_of_image(image_path):
    """Convert image to base64 string"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

def get_logo_html(size="medium"):
    """Get logo HTML with custom PNG or fallback to icon"""
    custom_logo_path = "logo.png"
    
    if os.path.exists(custom_logo_path):
        logo_base64 = get_base64_of_image(custom_logo_path)
        if logo_base64:
            sizes = {
                "small": "60px",
                "medium": "120px",
                "large": "180px",
                "xlarge": "300px",
                "sidebar": "160px",
                "header": "150px"
            }
            return f'<img src="data:image/png;base64,{logo_base64}" style="width: {sizes.get(size, "80px")}; height: auto; object-fit: contain;">'
    
    icon_urls = {
        "small": "https://img.icons8.com/fluency/48/000000/mental-state.png",
        "medium": "https://img.icons8.com/fluency/96/000000/mental-state.png",
        "large": "https://img.icons8.com/fluency/240/000000/mental-state.png",
        "xlarge": "https://img.icons8.com/fluency/480/000000/mental-state.png",
        "sidebar": "https://img.icons8.com/fluency/96/000000/mental-state.png",
        "header": "https://img.icons8.com/fluency/240/000000/mental-state.png",
        "login": "https://img.icons8.com/fluency/240/000000/mental-state.png"
    }
    
    sizes = {
        "small": "60px",
        "medium": "120px",
        "large": "180px",
        "xlarge": "300px",
        "sidebar": "100px",
        "header": "150px",
        "login": "200px"
    }
    return f'<img src="{icon_urls.get(size, icon_urls["medium"])}" style="width: {sizes.get(size, "80px")}; height: auto;">'

# ==================== USER AUTHENTICATION SYSTEM ====================
USER_DB_FILE = 'users.json'
PROFILE_IMAGES_DIR = 'profile_images'

if not os.path.exists(PROFILE_IMAGES_DIR):
    os.makedirs(PROFILE_IMAGES_DIR)

def init_user_db():
    """Initialize the user database file if it doesn't exist"""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'w') as f:
            json.dump({}, f)

def hash_password(password):
    """Hash a password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def save_profile_image(username, image_data):
    """Save profile image for user"""
    try:
        old_pattern = f"{PROFILE_IMAGES_DIR}/{username}_*"
        for old_file in Path(PROFILE_IMAGES_DIR).glob(f"{username}_*"):
            old_file.unlink()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"{PROFILE_IMAGES_DIR}/{username}_{timestamp}.png"
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        return True, image_path
    except Exception as e:
        return False, str(e)

def get_profile_image(username):
    """Get profile image path for user"""
    try:
        images = list(Path(PROFILE_IMAGES_DIR).glob(f"{username}_*"))
        if images:
            return str(images[-1])
    except:
        pass
    return None

def create_user(username, password, email):
    """Create a new user account"""
    try:
        if os.path.exists(USER_DB_FILE):
            with open(USER_DB_FILE, 'r') as f:
                users = json.load(f)
        else:
            users = {}
        
        if username in users:
            return False, "Username already exists!"
        
        users[username] = {
            'password': hash_password(password),
            'email': email,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'history': [],
            'settings': {
                'notifications': True,
                'save_history': True,
                'theme': 'Light'
            }
        }
        
        with open(USER_DB_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        
        return True, "Account created successfully!"
    
    except Exception as e:
        return False, f"Error creating account: {str(e)}"

def verify_user(username, password):
    """Verify user credentials"""
    try:
        if not os.path.exists(USER_DB_FILE):
            return False, "No users found. Please sign up first."
        
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        if username not in users:
            return False, "Invalid username or password"
        
        hashed_input = hash_password(password)
        stored_password = users[username]['password']
        
        if hashed_input == stored_password:
            return True, users[username]
        else:
            return False, "Invalid username or password"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

def save_analysis_to_history(username, analysis_data):
    """Save analysis results to user history"""
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        if username in users:
            if 'history' not in users[username]:
                users[username]['history'] = []
            
            if 'timestamp' not in analysis_data:
                analysis_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            users[username]['history'].append({
                'timestamp': analysis_data['timestamp'],
                'data': analysis_data
            })
            
            if len(users[username]['history']) > 50:
                users[username]['history'] = users[username]['history'][-50:]
            
            with open(USER_DB_FILE, 'w') as f:
                json.dump(users, f, indent=4)
            
            return True
    except Exception as e:
        print(f"Error saving to history: {e}")
        return False

def update_user_settings(username, settings):
    """Update user settings"""
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        if username in users:
            users[username]['settings'] = settings
            with open(USER_DB_FILE, 'w') as f:
                json.dump(users, f, indent=4)
            return True
    except Exception as e:
        print(f"Error updating settings: {e}")
        return False

def load_user_history(username):
    """Load user history from database"""
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        if username in users and 'history' in users[username]:
            history = users[username]['history']
            if isinstance(history, list):
                return history
            else:
                return []
    except Exception as e:
        print(f"Error loading history: {e}")
    return []

def migrate_history_format():
    """Migrate old history format to new format"""
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        modified = False
        for username, user_data in users.items():
            if 'history' in user_data and isinstance(user_data['history'], list):
                new_history = []
                for entry in user_data['history']:
                    if isinstance(entry, dict):
                        if 'data' in entry:
                            new_history.append(entry)
                        elif 'dominant_emotion' in entry:
                            new_history.append({
                                'timestamp': entry.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                                'data': entry
                            })
                            modified = True
                        else:
                            modified = True
                    else:
                        modified = True
                
                if modified:
                    users[username]['history'] = new_history
        
        if modified:
            with open(USER_DB_FILE, 'w') as f:
                json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Error during migration: {e}")

# ==================== LOGO CSS ====================
LOGO_CSS = """
<style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .pulse-logo {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .float-logo {
        animation: float 3s ease-in-out infinite;
    }
    
    .rotate-logo {
        animation: rotate 20s linear infinite;
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    .gradient-text {
        background: linear-gradient(45deg, #4A90E2, #9C27B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    
    .profile-image-container {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        overflow: hidden;
        border: 3px solid #4A90E2;
        margin: 0 auto 1rem auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .profile-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .sidebar-profile {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea20, #764ba220);
        border-radius: 15px;
        margin-bottom: 1rem;
        animation: slideIn 0.5s ease-out;
    }
    
    .sidebar-profile-image {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin: 0 auto 0.5rem auto;
        border: 2px solid #4A90E2;
        object-fit: cover;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea10, #764ba210);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #4A90E230;
        transition: transform 0.3s;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.2);
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    .waveform-container {
        background: linear-gradient(135deg, #667eea10, #764ba210);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .voice-meter {
        width: 100%;
        height: 10px;
        background: linear-gradient(90deg, #4A90E2, #9C27B0);
        border-radius: 5px;
        transition: width 0.3s;
    }
</style>
"""

# ==================== MAIN APP CONFIGURATION ====================
st.set_page_config(
    page_title="NeuroWell - Mental Health Emotion Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add logo CSS
st.markdown(LOGO_CSS, unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4A90E2;
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
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .wellness-tip {
        background-color: #E8F4FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin: 1rem 0;
    }
    .history-entry {
        background-color: #f9f9f9;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .stButton button {
        width: 100%;
    }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .upload-text {
        color: #4A90E2;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea10, #764ba210);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #4A90E230;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4A90E2;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    .crisis-critical {
        background-color: #7F1D1D20;
        border-left: 5px solid #7F1D1D;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .crisis-high {
        background-color: #EF444420;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .crisis-moderate {
        background-color: #F59E0B20;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .crisis-low {
        background-color: #FBBF2420;
        border-left: 5px solid #FBBF24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .voice-recording {
        background: linear-gradient(135deg, #667eea20, #764ba220);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        border: 2px dashed #4A90E2;
        margin: 1rem 0;
    }
    .recording-indicator {
        width: 20px;
        height: 20px;
        background-color: #EF4444;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        animation: pulse 1s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize user database and migrate history
init_user_db()
migrate_history_format()

# ==================== SESSION STATE INITIALIZATION ====================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'profile_image' not in st.session_state:
    st.session_state.profile_image = None
if 'current_text_analysis' not in st.session_state:
    st.session_state.current_text_analysis = None
if 'current_voice_analysis' not in st.session_state:
    st.session_state.current_voice_analysis = None
if 'recording' not in st.session_state:
    st.session_state.recording = False

# Initialize enhanced analyzers
voice_analyzer = EnhancedVoiceEmotionAnalyzer()
text_analyzer = EnhancedTextEmotionAnalyzer()

# ==================== MODEL LOADING ====================
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_emotion_model():
    model_path = 'emotiondetector.h5'
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        return None

model = load_emotion_model()

# ==================== CHART GENERATION FUNCTIONS ====================
def create_emotion_timeline(df):
    """Create timeline chart of emotions over time"""
    if df.empty:
        return None
    
    try:
        df_copy = df.copy()
        
        # Check if timestamp column exists
        if 'timestamp' not in df_copy.columns:
            return None
        
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
        df_copy = df_copy.dropna(subset=['datetime'])
        
        if df_copy.empty:
            return None
        
        df_copy = df_copy.sort_values('datetime')
        
        fig = go.Figure()
        
        for emotion in CLASS_NAMES:
            emotion_data = df_copy[df_copy['dominant_emotion'] == emotion]
            if not emotion_data.empty:
                fig.add_trace(go.Scatter(
                    x=emotion_data['datetime'],
                    y=[1] * len(emotion_data),
                    mode='markers+lines',
                    name=emotion.capitalize(),
                    marker=dict(
                        size=10,
                        color=MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#808080'),
                        symbol='circle'
                    ),
                    line=dict(
                        color=MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#808080'),
                        width=1,
                        dash='dot'
                    )
                ))
        
        fig.update_layout(
            title="Emotion Timeline",
            xaxis_title="Date & Time",
            yaxis_title="Emotion Occurrence",
            showlegend=True,
            height=400,
            hovermode='closest',
            yaxis=dict(
                showticklabels=False,
                showgrid=False
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error creating timeline: {e}")
        return None

def create_weekly_pattern(df):
    """Create weekly pattern chart"""
    if df.empty:
        return None
    
    try:
        df_copy = df.copy()
        
        # Check if timestamp column exists
        if 'timestamp' not in df_copy.columns:
            return None
        
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
        df_copy = df_copy.dropna(subset=['datetime'])
        
        if df_copy.empty:
            return None
        
        df_copy['day'] = df_copy['datetime'].dt.day_name()
        
        # Check if dominant_emotion column exists
        if 'dominant_emotion' not in df_copy.columns:
            return None
        
        day_emotion = pd.crosstab(df_copy['day'], df_copy['dominant_emotion'])
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_emotion = day_emotion.reindex(day_order, fill_value=0)
        
        fig = go.Figure()
        
        for emotion in CLASS_NAMES:
            if emotion in day_emotion.columns:
                fig.add_trace(go.Bar(
                    name=emotion.capitalize(),
                    x=day_emotion.index,
                    y=day_emotion[emotion],
                    marker_color=MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#808080')
                ))
        
        fig.update_layout(
            title="Weekly Emotion Pattern",
            xaxis_title="Day of Week",
            yaxis_title="Count",
            barmode='stack',
            height=400
        )
        
        return fig
    except Exception as e:
        print(f"Error creating weekly pattern: {e}")
        return None

def create_emotion_radar(df):
    """Create radar chart of emotion averages"""
    if df.empty:
        return None
    
    try:
        # Check if required columns exist
        if 'dominant_emotion' not in df.columns or 'confidence' not in df.columns:
            return None
        
        emotion_avgs = []
        for emotion in CLASS_NAMES:
            emotion_data = df[df['dominant_emotion'] == emotion]
            if not emotion_data.empty:
                avg_conf = emotion_data['confidence'].mean() * 100
            else:
                avg_conf = 0
            emotion_avgs.append(avg_conf)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=emotion_avgs,
            theta=[e.capitalize() for e in CLASS_NAMES],
            fill='toself',
            marker=dict(
                color=[MENTAL_HEALTH_MAPPING.get(e, {}).get('color', '#808080') for e in CLASS_NAMES]
            ),
            line=dict(color='#4A90E2', width=2)
        ))
        
        fig.update_layout(
            title="Emotion Profile Radar",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            height=400
        )
        
        return fig
    except Exception as e:
        print(f"Error creating radar: {e}")
        return None

# ==================== PROFESSIONAL PDF REPORT GENERATION ====================
class ProfessionalPDF(FPDF):
    """Enhanced PDF class for professional hospital-style reports"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)
        
    def header(self):
        self.set_fill_color(0, 94, 184)
        self.rect(0, 0, 210, 40, 'F')
        
        self.set_y(10)
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'NEUROWELL MENTAL HEALTH CENTER', 0, 1, 'C')
        
        self.set_font('Helvetica', '', 12)
        self.set_text_color(230, 230, 230)
        self.cell(0, 8, 'Emotional Wellness Assessment Report', 0, 1, 'C')
        
        self.set_y(42)
        self.set_draw_color(255, 184, 28)
        self.set_line_width(1)
        self.line(10, 42, 200, 42)
        self.ln(15)
    
    def footer(self):
        self.set_y(-20)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, f'Page {self.page_no()} | Generated by NeuroWell AI System', 0, 0, 'C')
        self.set_draw_color(0, 94, 184)
        self.line(10, 275, 200, 275)
    
    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 94, 184)
        self.cell(0, 10, self._clean_text(title), 0, 1, 'L')
        self.set_draw_color(0, 94, 184)
        self.line(self.get_x(), self.get_y(), 200, self.get_y())
        self.ln(5)
    
    def _clean_text(self, text):
        """Remove or replace special characters for PDF compatibility"""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        replacements = {
            '•': '-', '●': '-', '○': 'o', '◆': '*', '▪': '-',
            '✓': '+', '✗': 'X', '★': '*', '☆': '*', '♥': '<3',
            '♦': '<>', '♣': 'C', '♠': 'S', '→': '->', '←': '<-',
            '↑': '^', '↓': 'v', '©': '(C)', '®': '(R)', '™': '(TM)',
            '°': ' degrees', '±': '+/-', '×': 'x', '÷': '/',
            '€': 'EUR', '£': 'GBP', '¥': 'JPY', '₹': 'INR',
            '«': '"', '»': '"', '—': '-', '–': '-', '…': '...',
            '“': '"', '”': '"', '‘': "'", '’': "'",
            '\u2022': '-', '\u2023': '-', '\u25CF': '-',
            '\u25CB': 'o', '\u25A0': '-', '\u25A1': '[]'
        }
        
        emoji_replacements = {
            '😊': ':happy:', '😠': ':angry:', '😢': ':sad:',
            '😨': ':fear:', '😐': ':neutral:', '😲': ':surprise:',
            '🤢': ':disgust:', '😤': ':frustrated:', '😰': ':anxious:',
            '😔': ':sad:', '😱': ':terrified:', '😡': ':furious:',
            '🥺': ':pleading:', '🤔': ':thoughtful:', '😌': ':relieved:',
            '😎': ':cool:', '🥳': ':celebrating:', '😴': ':sleepy:',
            '🤒': ':sick:', '🤕': ':hurt:', '🥴': ':dizzy:',
            '🤯': ':mind-blown:', '😬': ':nervous:', '🥲': ':tearful:',
            '😶‍🌫️': ':foggy:', '😮‍💨': ':exhaling:', '😵‍💫': ':dizzy:',
            '😳': ':embarrassed:'
        }
        
        for old, new in {**replacements, **emoji_replacements}.items():
            text = text.replace(old, new)
        
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text
    
    def patient_info_card(self, username, start_date, end_date, total):
        self.set_fill_color(245, 245, 245)
        self.rect(10, self.get_y(), 190, 40, 'F')
        
        self.set_xy(15, self.get_y() + 5)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 94, 184)
        self.cell(40, 6, 'Patient Name:', 0, 0)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(60, 6, self._clean_text(username), 0, 1)
        
        self.set_x(15)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 94, 184)
        self.cell(40, 6, 'Medical ID:', 0, 0)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(60, 6, f'NW-{abs(hash(username)) % 10000:04d}', 0, 1)
        
        self.set_x(15)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 94, 184)
        self.cell(40, 6, 'Assessment Period:', 0, 0)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(60, 6, f'{start_date} to {end_date}', 0, 1)
        
        self.set_x(120)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 94, 184)
        self.cell(40, 6, 'Total Sessions:', 0, 0)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(30, 6, str(total), 0, 1)
        
        self.ln(10)
    
    def metric_box(self, x, y, label, value, color):
        self.set_xy(x, y)
        self.set_fill_color(color[0], color[1], color[2])
        self.rect(x, y, 45, 25, 'F')
        
        self.set_xy(x, y + 5)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.cell(45, 8, self._clean_text(str(value)), 0, 1, 'C')
        
        self.set_xy(x, y + 13)
        self.set_font('Helvetica', '', 8)
        self.cell(45, 8, self._clean_text(label), 0, 1, 'C')
    
    def create_table(self, headers, data, col_widths):
        self.set_fill_color(0, 94, 184)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 10)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, self._clean_text(header), 1, 0, 'C', 1)
        self.ln()
        
        self.set_text_color(0, 0, 0)
        self.set_font('Helvetica', '', 9)
        fill = False
        
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 8, self._clean_text(str(item)), 'LR', 0, 'L' if i == 0 else 'C', fill)
            self.ln()
            fill = not fill
        
        self.cell(sum(col_widths), 0, '', 'T')

# ==================== TEXT PDF REPORT GENERATION ====================
def generate_text_pdf_report(text_data, username, start_date, end_date, title):
    """Generate a PDF report specifically for text analysis data"""
    
    pdf = ProfessionalPDF()
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(0, 94, 184)
    pdf.cell(0, 15, pdf._clean_text(title), 0, 1, 'C')
    pdf.ln(5)
    
    pdf.patient_info_card(username, start_date, end_date, len(text_data))
    
    pdf.section_title('TEXT ANALYSIS SUMMARY')
    
    # Calculate metrics
    total_texts = len(text_data)
    
    # Safely calculate averages
    sentiment_sum = 0
    word_count_sum = 0
    crisis_count = 0
    
    for entry in text_data:
        # Sentiment
        if 'sentiment' in entry and isinstance(entry['sentiment'], dict):
            if 'combined' in entry['sentiment']:
                sentiment_sum += entry['sentiment']['combined'].get('compound', 0)
        
        # Word count
        if 'statistics' in entry and isinstance(entry['statistics'], dict):
            word_count_sum += entry['statistics'].get('word_count', 0)
        
        # Crisis
        if 'crisis_risk' in entry and isinstance(entry['crisis_risk'], dict):
            if entry['crisis_risk'].get('risk_level', 'none') != 'none':
                crisis_count += 1
    
    avg_sentiment = sentiment_sum / total_texts if total_texts > 0 else 0
    avg_word_count = word_count_sum / total_texts if total_texts > 0 else 0
    
    # Get emotion distribution
    emotions = []
    for entry in text_data:
        if 'dominant_emotion' in entry:
            emotions.append(entry['dominant_emotion'])
    
    emotion_counts = Counter(emotions)
    
    y = pdf.get_y()
    pdf.metric_box(10, y, 'Total Texts', total_texts, (0, 94, 184))
    pdf.metric_box(60, y, 'Avg Sentiment', f'{avg_sentiment:.2f}', (99, 102, 241))
    pdf.metric_box(110, y, 'Avg Words', f'{avg_word_count:.0f}', (34, 197, 94))
    pdf.metric_box(160, y, 'Crisis Alerts', crisis_count, (239, 68, 68))
    
    pdf.ln(30)
    
    pdf.section_title('EMOTION DISTRIBUTION')
    
    headers = ['Emotion', 'Count', 'Percentage']
    col_widths = [60, 40, 60]
    
    data = []
    for emotion, count in emotion_counts.items():
        percentage = (count / total_texts) * 100 if total_texts > 0 else 0
        data.append([emotion.capitalize(), str(count), f'{percentage:.1f}%'])
    
    # Add missing emotions
    all_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'anticipation', 'neutral']
    for emotion in all_emotions:
        if emotion not in emotion_counts:
            data.append([emotion.capitalize(), '0', '0.0%'])
    
    data.sort(key=lambda x: x[0])
    pdf.create_table(headers, data, col_widths)
    pdf.ln(15)
    
    pdf.section_title('RECENT TEXT ANALYSES')
    
    headers = ['Date', 'Primary Emotion', 'Sentiment', 'Risk']
    col_widths = [45, 35, 35, 35]
    
    recent_data = []
    sorted_entries = sorted(text_data, key=lambda x: x.get('timestamp', ''), reverse=True)[:8]
    
    for entry in sorted_entries:
        date = entry.get('timestamp', '')[:10] if entry.get('timestamp') else 'N/A'
        emotion = entry.get('dominant_emotion', 'unknown').capitalize()
        sentiment = f"{entry.get('sentiment', {}).get('combined', {}).get('compound', 0):.2f}" if isinstance(entry.get('sentiment'), dict) else 'N/A'
        risk = entry.get('crisis_risk', {}).get('risk_level', 'none').upper() if isinstance(entry.get('crisis_risk'), dict) else 'NONE'
        recent_data.append([date, emotion, sentiment, risk])
    
    pdf.create_table(headers, recent_data, col_widths)
    pdf.ln(15)
    
    # Sample texts section
    pdf.section_title('TEXT SAMPLES')
    
    for i, entry in enumerate(sorted_entries[:3]):  # Show only 3 samples
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(0, 94, 184)
        pdf.cell(0, 8, f'Sample {i+1}:', 0, 1)
        
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(66, 85, 99)
        
        text_preview = entry.get('text_preview', 'No preview available')
        if len(text_preview) > 100:
            text_preview = text_preview[:100] + '...'
        
        pdf.multi_cell(0, 5, pdf._clean_text(text_preview))
        pdf.ln(3)
    
    # Add crisis resources if any crises detected
    if crisis_count > 0:
        pdf.add_page()
        pdf.section_title('CRISIS RESOURCES')
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(239, 68, 68)
        pdf.cell(0, 10, '⚠️ IMPORTANT: Crisis Indicators Detected', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(66, 85, 99)
        pdf.multi_cell(0, 6, 'The following resources are available 24/7:')
        pdf.ln(5)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'National Suicide Prevention Lifeline:', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, '988 or 1-800-273-8255', 0, 1)
        pdf.ln(3)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'Crisis Text Line:', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, 'Text HOME to 741741', 0, 1)
        pdf.ln(3)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'SAMHSA National Helpline:', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, '1-800-662-4357', 0, 1)
        pdf.ln(3)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'Emergency:', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, '911', 0, 1)
        pdf.ln(10)
    
    # Confidentiality Notice
    pdf.set_y(-50)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, 'CONFIDENTIALITY NOTICE: This report contains protected health information. Unauthorized access or distribution is prohibited. This document is electronically generated and valid without signature.', 0, 'C')
    
    # Signatures
    pdf.set_y(-35)
    pdf.set_font('Helvetica', '', 9)
    pdf.line(20, pdf.get_y(), 90, pdf.get_y())
    pdf.line(120, pdf.get_y(), 190, pdf.get_y())
    pdf.set_xy(20, pdf.get_y() + 2)
    pdf.cell(70, 5, 'Clinical Psychologist', 0, 0, 'C')
    pdf.set_x(120)
    pdf.cell(70, 5, 'Patient', 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1', errors='ignore')

# ==================== VOICE PDF REPORT GENERATION ====================
def generate_voice_pdf_report(voice_data, username, start_date, end_date, title):
    """Generate a PDF report specifically for voice analysis data"""
    
    pdf = ProfessionalPDF()
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(0, 94, 184)
    pdf.cell(0, 15, pdf._clean_text(title), 0, 1, 'C')
    pdf.ln(5)
    
    pdf.patient_info_card(username, start_date, end_date, len(voice_data))
    
    pdf.section_title('VOICE ANALYSIS SUMMARY')
    
    # Calculate metrics
    total_recordings = len(voice_data)
    
    # Emotion distribution
    emotions = []
    for entry in voice_data:
        if 'dominant_emotion' in entry:
            emotions.append(entry['dominant_emotion'])
    
    emotion_counts = Counter(emotions)
    
    # Average confidence
    conf_sum = sum([e.get('confidence', 0) for e in voice_data])
    avg_conf = (conf_sum / total_recordings * 100) if total_recordings > 0 else 0
    
    # Voice health issues
    health_issues = 0
    for entry in voice_data:
        if 'health_indicators' in entry and entry['health_indicators']:
            health_issues += len(entry['health_indicators'])
    
    y = pdf.get_y()
    pdf.metric_box(10, y, 'Total Recordings', total_recordings, (0, 94, 184))
    pdf.metric_box(60, y, 'Avg Confidence', f'{avg_conf:.1f}%', (99, 102, 241))
    pdf.metric_box(110, y, 'Unique Emotions', len(emotion_counts), (34, 197, 94))
    pdf.metric_box(160, y, 'Health Alerts', health_issues, (239, 68, 68))
    
    pdf.ln(30)
    
    pdf.section_title('VOICE EMOTION DISTRIBUTION')
    
    headers = ['Emotion', 'Count', 'Percentage', 'Avg Confidence']
    col_widths = [45, 30, 40, 45]
    
    data = []
    for emotion, count in emotion_counts.items():
        percentage = (count / total_recordings) * 100
        # Get avg confidence for this emotion
        emotion_entries = [e for e in voice_data if e.get('dominant_emotion') == emotion]
        avg_emotion_conf = sum([e.get('confidence', 0) for e in emotion_entries]) / len(emotion_entries) * 100
        data.append([
            emotion.capitalize(), 
            str(count), 
            f'{percentage:.1f}%',
            f'{avg_emotion_conf:.1f}%'
        ])
    
    # Add missing emotions
    all_emotions = list(VOICE_EMOTION_MAPPING.keys())
    for emotion in all_emotions:
        if emotion not in emotion_counts:
            data.append([emotion.capitalize(), '0', '0.0%', '0.0%'])
    
    data.sort(key=lambda x: x[0])
    pdf.create_table(headers, data, col_widths)
    pdf.ln(15)
    
    pdf.section_title('RECENT VOICE ANALYSES')
    
    headers = ['Date', 'Emotion', 'Confidence', 'Pitch (Hz)']
    col_widths = [45, 30, 30, 45]
    
    recent_data = []
    sorted_entries = sorted(voice_data, key=lambda x: x.get('timestamp', ''), reverse=True)[:8]
    
    for entry in sorted_entries:
        date = entry.get('timestamp', '')[:10] if entry.get('timestamp') else 'N/A'
        emotion = entry.get('dominant_emotion', 'unknown').capitalize()
        confidence = f"{entry.get('confidence', 0)*100:.1f}%"
        pitch = f"{entry.get('features', {}).get('pitch_mean', 0):.0f}"
        recent_data.append([date, emotion, confidence, pitch])
    
    pdf.create_table(headers, recent_data, col_widths)
    pdf.ln(15)
    
    # Voice Health Recommendations
    if health_issues > 0:
        pdf.section_title('VOICE HEALTH RECOMMENDATIONS')
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(239, 68, 68)
        pdf.cell(0, 8, '⚠️ Voice Health Alerts Detected', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(66, 85, 99)
        
        recommendations = set()
        for entry in voice_data:
            if 'health_indicators' in entry:
                for indicator in entry['health_indicators']:
                    recommendations.add(indicator['suggestion'])
        
        for rec in recommendations:
            pdf.multi_cell(0, 5, f'• {pdf._clean_text(rec)}')
            pdf.ln(2)
        
        pdf.ln(5)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(0, 94, 184)
        pdf.cell(0, 8, 'Voice Care Tips:', 0, 1)
        
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, '• Stay hydrated by drinking plenty of water')
        pdf.multi_cell(0, 5, '• Avoid shouting or straining your voice')
        pdf.multi_cell(0, 5, '• Take vocal rest when feeling tired')
        pdf.multi_cell(0, 5, '• Practice deep breathing exercises')
        pdf.ln(5)
    
    # Confidentiality Notice
    pdf.set_y(-50)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, 'CONFIDENTIALITY NOTICE: This report contains protected health information. Unauthorized access or distribution is prohibited. This document is electronically generated and valid without signature.', 0, 'C')
    
    # Signatures
    pdf.set_y(-35)
    pdf.set_font('Helvetica', '', 9)
    pdf.line(20, pdf.get_y(), 90, pdf.get_y())
    pdf.line(120, pdf.get_y(), 190, pdf.get_y())
    pdf.set_xy(20, pdf.get_y() + 2)
    pdf.cell(70, 5, 'Speech Language Pathologist', 0, 0, 'C')
    pdf.set_x(120)
    pdf.cell(70, 5, 'Patient', 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1', errors='ignore')

# ==================== FACIAL ANALYSIS PDF REPORT GENERATION ====================
def generate_professional_pdf_report(df, username, start_date, end_date, title):
    """Generate a professional hospital-style PDF report for facial analysis data"""
    
    pdf = ProfessionalPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(0, 94, 184)
    pdf.cell(0, 15, pdf._clean_text(title), 0, 1, 'C')
    pdf.ln(5)
    
    # Patient Information Card
    pdf.patient_info_card(username, start_date, end_date, len(df))
    
    # Clinical Summary Section
    pdf.section_title('CLINICAL SUMMARY')
    
    # Calculate metrics
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    if 'dominant_emotion' in df.columns:
        positive_count = len(df[df['dominant_emotion'].isin(
            [e for e in CLASS_NAMES if MENTAL_HEALTH_MAPPING.get(e, {}).get('category') == 'Positive']
        )])
        negative_count = len(df[df['dominant_emotion'].isin(
            [e for e in CLASS_NAMES if MENTAL_HEALTH_MAPPING.get(e, {}).get('category') == 'Negative']
        )])
        neutral_count = len(df[df['dominant_emotion'].isin(
            [e for e in CLASS_NAMES if MENTAL_HEALTH_MAPPING.get(e, {}).get('category') == 'Neutral']
        )])
    
    avg_confidence = df['confidence'].mean() * 100 if 'confidence' in df.columns else 0
    emotional_diversity = df['dominant_emotion'].nunique() if 'dominant_emotion' in df.columns else 0
    most_common = df['dominant_emotion'].mode()[0] if not df.empty and 'dominant_emotion' in df.columns else "neutral"
    
    # Wellness score calculation
    total = len(df)
    positive_ratio = positive_count / total if total > 0 else 0
    negative_ratio = negative_count / total if total > 0 else 0
    wellness_score = (positive_ratio * 70 + (1 - negative_ratio) * 30) * (avg_confidence / 100) * 100
    
    if wellness_score >= 80:
        wellness_category = "EXCELLENT"
    elif wellness_score >= 60:
        wellness_category = "GOOD"
    elif wellness_score >= 40:
        wellness_category = "FAIR"
    else:
        wellness_category = "ATTENTION NEEDED"
    
    # Metrics boxes
    y = pdf.get_y()
    pdf.metric_box(10, y, 'Total Sessions', len(df), (0, 94, 184))
    pdf.metric_box(60, y, 'Positive', positive_count, (34, 197, 94))
    pdf.metric_box(110, y, 'Challenging', negative_count, (239, 68, 68))
    pdf.metric_box(160, y, 'Neutral', neutral_count, (107, 114, 128))
    
    pdf.ln(30)
    
    # Additional metrics
    y = pdf.get_y()
    pdf.metric_box(10, y, 'Avg Confidence', f'{avg_confidence:.1f}%', (99, 102, 241))
    pdf.metric_box(60, y, 'Diversity', f'{emotional_diversity}/7', (245, 158, 11))
    pdf.metric_box(110, y, 'Most Common', most_common.capitalize(), (139, 92, 246))
    pdf.metric_box(160, y, 'Wellness', wellness_category, (0, 94, 184))
    
    pdf.ln(30)
    
    # Clinical Interpretation
    pdf.section_title('CLINICAL INTERPRETATION')
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(66, 85, 99)
    
    interpretation = f"""
    Based on {len(df)} clinical assessments, the patient demonstrates a predominantly {most_common.capitalize()} 
    emotional state. The emotional distribution shows {positive_count} positive ({positive_ratio*100:.1f}%), 
    {negative_count} challenging ({negative_ratio*100:.1f}%), and {neutral_count} neutral states.
    
    The overall wellness score indicates a {wellness_category.lower()} emotional balance.
    """
    pdf.multi_cell(0, 6, pdf._clean_text(interpretation))
    pdf.ln(5)
    
    if negative_ratio > 0.5:
        recommendation = "Clinical Recommendation: Elevated challenging emotions detected. Recommend therapeutic intervention and stress management techniques."
    elif positive_ratio > 0.7:
        recommendation = "Clinical Recommendation: Excellent emotional resilience. Maintain current coping strategies and consider peer support opportunities."
    else:
        recommendation = "Clinical Recommendation: Balanced emotional state within expected parameters. Continue monitoring and maintain wellness practices."
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(0, 94, 184)
    pdf.cell(0, 8, pdf._clean_text(recommendation), 0, 1)
    pdf.ln(10)
    
    # Emotion Distribution Table
    pdf.section_title('EMOTIONAL STATE DISTRIBUTION')
    
    headers = ['Emotion', 'Count', 'Percentage', 'Clinical Significance']
    col_widths = [45, 30, 40, 75]
    
    data = []
    if 'dominant_emotion' in df.columns:
        emotion_counts = df['dominant_emotion'].value_counts()
        total = len(df)
        
        for emotion, count in emotion_counts.items():
            percentage = (count / total) * 100
            if percentage > 40:
                significance = "Clinically significant"
            elif percentage > 20:
                significance = "Moderate presence"
            else:
                significance = "Within normal range"
            data.append([emotion.capitalize(), str(count), f'{percentage:.1f}%', significance])
        
        # Add missing emotions
        for emotion in CLASS_NAMES:
            if emotion not in emotion_counts.index:
                data.append([emotion.capitalize(), '0', '0.0%', 'Not observed'])
        
        data.sort(key=lambda x: x[0])
        pdf.create_table(headers, data, col_widths)
    else:
        pdf.cell(0, 10, "No emotion data available", 0, 1)
    
    pdf.ln(15)
    
    # Recent Activity
    pdf.section_title('RECENT ASSESSMENT LOG')
    
    headers = ['Date/Time', 'Emotion', 'Confidence', 'Status']
    col_widths = [45, 30, 30, 45]
    
    recent_data = []
    if 'timestamp' in df.columns and 'dominant_emotion' in df.columns:
        # Create datetime column for sorting
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
        recent_df = df_copy.sort_values('datetime', ascending=False).head(8)
        
        for _, row in recent_df.iterrows():
            timestamp = row['datetime'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['datetime']) else str(row['timestamp'])[:16]
            emotion = row['dominant_emotion'].capitalize()
            confidence = f"{row['confidence']*100:.1f}%" if 'confidence' in row else "N/A"
            status = MENTAL_HEALTH_MAPPING.get(row['dominant_emotion'], {}).get('status', '')
            status = pdf._clean_text(status)[:20]
            recent_data.append([timestamp, emotion, confidence, status])
        
        pdf.create_table(headers, recent_data, col_widths)
    else:
        pdf.cell(0, 10, "No recent activity data available", 0, 1)
    
    pdf.ln(15)
    
    # Treatment Recommendations
    pdf.section_title('TREATMENT RECOMMENDATIONS')
    
    most_common_data = MENTAL_HEALTH_MAPPING.get(most_common, MENTAL_HEALTH_MAPPING['neutral'])
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(0, 94, 184)
    pdf.cell(0, 8, f'Primary Recommendation (based on {most_common.capitalize()} predominance):', 0, 1)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(66, 85, 99)
    
    activities = [pdf._clean_text(act) for act in most_common_data["activities"]]
    intervention = pdf._clean_text(most_common_data.get("intervention", "Cognitive behavioral therapy"))
    
    pdf.multi_cell(0, 6, f'- Clinical Intervention: {intervention}')
    pdf.multi_cell(0, 6, f'- Daily Practice: {activities[0]}')
    pdf.multi_cell(0, 6, f'- Weekly Goal: {activities[1]}')
    pdf.multi_cell(0, 6, f'- As Needed: {activities[2]}')
    pdf.ln(5)
    
    # Wellness Prescription
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(0, 94, 184)
    pdf.cell(0, 8, 'Wellness Prescription:', 0, 1)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, f'1. Practice {activities[0].lower()} daily for emotional regulation')
    pdf.multi_cell(0, 6, f'2. Engage in {activities[1].lower()} 3-4 times per week')
    pdf.multi_cell(0, 6, '3. Maintain emotional awareness journal')
    pdf.multi_cell(0, 6, '4. Schedule follow-up assessment in 2 weeks')
    pdf.multi_cell(0, 6, '5. Contact provider if acute distress occurs')
    pdf.ln(10)
    
    # Confidentiality Notice
    pdf.set_y(-50)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, 'CONFIDENTIALITY NOTICE: This report contains protected health information. Unauthorized access or distribution is prohibited. This document is electronically generated and valid without signature.', 0, 'C')
    
    # Signatures
    pdf.set_y(-35)
    pdf.set_font('Helvetica', '', 9)
    pdf.line(20, pdf.get_y(), 90, pdf.get_y())
    pdf.line(120, pdf.get_y(), 190, pdf.get_y())
    pdf.set_xy(20, pdf.get_y() + 2)
    pdf.cell(70, 5, 'Clinical Psychologist', 0, 0, 'C')
    pdf.set_x(120)
    pdf.cell(70, 5, 'Patient', 0, 1, 'C')
    
    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1', errors='ignore')

# ==================== AUTHENTICATION UI ====================
def show_login_signup():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; animation: float 3s ease-in-out infinite; margin-bottom: 2rem;">
            {get_logo_html('large')}
            <h1 style="color: #4A90E2; font-size: 3rem; margin: 0.5rem 0 0 0; 
                       background: linear-gradient(45deg, #4A90E2, #9C27B0);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                NeuroWell
            </h1>
            <p style="color: #666; font-size: 1.2rem; margin: 0;">
                Mental Health & Emotion Companion
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted:
                    if username and password:
                        with st.spinner("Verifying..."):
                            success, result = verify_user(username, password)
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                profile_path = get_profile_image(username)
                                if profile_path:
                                    st.session_state.profile_image = profile_path
                                db_history = load_user_history(username)
                                if db_history:
                                    st.session_state.analysis_history = []
                                    for entry in db_history:
                                        if isinstance(entry, dict):
                                            if 'data' in entry:
                                                st.session_state.analysis_history.append(entry['data'])
                                            elif 'dominant_emotion' in entry:
                                                st.session_state.analysis_history.append(entry)
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error(result)
                    else:
                        st.warning("Please fill in all fields")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_email = st.text_input("Email", placeholder="Enter your email")
                new_password = st.text_input("Password", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                submitted = st.form_submit_button("Create Account", use_container_width=True)
                
                if submitted:
                    if new_username and new_email and new_password and confirm_password:
                        if new_password == confirm_password:
                            if len(new_password) >= 6:
                                with st.spinner("Creating account..."):
                                    success, message = create_user(new_username, new_password, new_email)
                                    if success:
                                        st.success(message)
                                        st.balloons()
                                        st.info("✅ Please login with your new account")
                                    else:
                                        st.error(message)
                            else:
                                st.error("Password must be at least 6 characters long")
                        else:
                            st.error("❌ Passwords don't match")
                    else:
                        st.warning("Please fill in all fields")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== ENHANCED VOICE ANALYSIS FUNCTIONS ====================
def show_voice_analysis_enhanced():
    """Enhanced voice analysis UI"""
    st.markdown("## 🎤 Enhanced Voice Emotion Analysis")
    st.markdown("Record or upload your voice for advanced emotion detection and health analysis")
    
    tab1, tab2 = st.tabs(["🎙️ Record Voice", "📁 Upload Audio"])
    
    with tab1:
        st.markdown("""
        <div class="voice-recording">
            <h3>Record Your Voice</h3>
            <p>Click the microphone below and speak naturally for 5-10 seconds.</p>
            <p class="small">For best results: Speak clearly, minimize background noise, and use a good microphone.</p>
        </div>
        """, unsafe_allow_html=True)
        
        audio_bytes = st.audio_input("Record a voice message")
        
        if audio_bytes:
            with st.spinner("Analyzing voice with enhanced ML algorithms..."):
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes.read())
                    tmp_path = tmp_file.name
                
                # Analyze voice with enhanced method
                result = analyze_voice_audio_enhanced(tmp_path)
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                if result:
                    display_voice_analysis_results_enhanced(result)
                    
                    # Save to history
                    if st.button("💾 Save to History", key="save_voice_recording"):
                        st.session_state.analysis_history.append(result)
                        if save_analysis_to_history(st.session_state.username, result):
                            st.success("✅ Voice analysis saved to your history!")
                            st.balloons()
                else:
                    st.error("Failed to analyze voice. Please ensure the audio is clear and at least 2 seconds long.")
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload an audio file", 
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="Supported formats: WAV, MP3, M4A, OGG, FLAC"
        )
        
        if uploaded_file is not None:
            with st.spinner("Analyzing voice with enhanced ML algorithms..."):
                # Save uploaded file to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Analyze voice with enhanced method
                result = analyze_voice_audio_enhanced(tmp_path)
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                if result:
                    display_voice_analysis_results_enhanced(result)
                    
                    # Save to history
                    if st.button("💾 Save to History", key="save_voice_upload"):
                        st.session_state.analysis_history.append(result)
                        if save_analysis_to_history(st.session_state.username, result):
                            st.success("✅ Voice analysis saved to your history!")
                            st.balloons()
                else:
                    st.error("Failed to analyze voice. Please ensure the audio file is valid and contains clear speech.")
    
    with st.expander("📋 Tips for Best Results"):
        st.markdown("""
        ### 🎯 For Accurate Voice Analysis:
        - **Speak clearly** at a natural pace
        - **Record for 5-10 seconds** (minimum 3 seconds)
        - **Minimize background noise**
        - **Use a good quality microphone**
        - **Speak naturally** - don't force emotions
        - **Ensure the audio is not too loud or too soft**
        - **Avoid background music or other voices**
        
        ### 🎵 Supported Audio Formats:
        - WAV (recommended)
        - MP3
        - M4A
        - OGG
        - FLAC
        """)

def analyze_voice_audio_enhanced(audio_path):
    """Enhanced voice analysis with better accuracy"""
    try:
        # Extract enhanced features
        features, y, sr = voice_analyzer.extract_enhanced_features(audio_path)
        
        if features is None:
            return None
        
        # Detect emotion with ML approach
        dominant_emotion, confidence, all_emotions = voice_analyzer.detect_emotion_ml(features)
        
        # Analyze voice health with advanced detection
        health_indicators = voice_analyzer.analyze_voice_health_advanced(features)
        
        # Get emotion data
        emotion_data = VOICE_EMOTION_MAPPING.get(dominant_emotion, VOICE_EMOTION_MAPPING['neutral'])
        
        # Calculate additional metrics
        duration = len(y) / sr
        
        # Calculate voice quality score
        quality_score = 0
        if features.get('zcr_mean', 0) < 0.15:
            quality_score += 0.3
        if features.get('pitch_std', 0) > 30:
            quality_score += 0.3
        if features.get('energy_mean', 0) > 0.03:
            quality_score += 0.4
            
        # Create result dictionary
        result = {
            'type': 'voice_analysis',
            'dominant_emotion': dominant_emotion,
            'confidence': confidence,
            'all_emotions': all_emotions,
            'features': features,
            'health_indicators': health_indicators,
            'emotion_data': emotion_data,
            'duration': duration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'quality_score': quality_score,
            'voice_quality': {
                'clarity': 'Excellent' if features.get('zcr_mean', 0) < 0.08 else 'Good' if features.get('zcr_mean', 0) < 0.15 else 'Moderate',
                'stability': 'Very Stable' if features.get('pitch_std', 100) < 30 else 'Stable' if features.get('pitch_std', 100) < 60 else 'Variable',
                'energy_level': 'High' if features.get('energy_mean', 0) > 0.15 else 'Moderate' if features.get('energy_mean', 0) > 0.05 else 'Low'
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error in voice analysis: {str(e)}")
        return None

def display_voice_analysis_results_enhanced(result):
    """Enhanced display with more details"""
    
    st.markdown("---")
    st.markdown("## 🎤 Enhanced Voice Analysis Results")
    
    # Main emotion display with confidence
    emotion_data = result['emotion_data']
    dominant_emotion = result['dominant_emotion']
    confidence = result['confidence']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        confidence_color = '#22C55E' if confidence > 0.8 else '#F59E0B' if confidence > 0.6 else '#EF4444'
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {emotion_data['color']}20, {emotion_data['color']}40); 
                    border-radius: 20px; border: 2px solid {emotion_data['color']};">
            <h1 style="font-size: 4rem; margin: 0;">{emotion_data['icon']}</h1>
            <h2 style="color: {emotion_data['color']}; margin: 0.5rem 0;">{dominant_emotion.upper()}</h2>
            <p style="font-size: 1.2rem; margin: 0;">Confidence: <span style="color: {confidence_color};">{confidence*100:.1f}%</span></p>
            <div style="width: 100%; background-color: #E0E0E0; border-radius: 10px; margin-top: 1rem;">
                <div style="width: {confidence*100}%; height: 10px; background: linear-gradient(90deg, {emotion_data['color']}, {confidence_color}); 
                            border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding: 1rem;">
            <h3 style="color: {emotion_data['color']};">{emotion_data['status']}</h3>
            <p><strong>Clinical Term:</strong> {emotion_data.get('clinical_term', 'N/A')}</p>
            <p><strong>Suggestion:</strong> {emotion_data['suggestion']}</p>
            <p><strong>Wellness Tip:</strong> {emotion_data['wellness_tip']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Voice Quality Metrics
    st.markdown("### 🎵 Voice Quality Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{result['duration']:.1f}s")
    with col2:
        st.metric("Avg Pitch", f"{result['features'].get('pitch_mean', 0):.0f} Hz", 
                 delta=f"±{result['features'].get('pitch_std', 0):.0f} Hz")
    with col3:
        energy = result['features'].get('energy_mean', 0)
        st.metric("Energy Level", f"{energy:.3f}", 
                 delta="High" if energy > 0.15 else "Low" if energy < 0.05 else "Moderate")
    with col4:
        st.metric("Speech Rate", f"{result['features'].get('speech_rate', 0):.0f} spm")
    
    # Voice Quality Indicators
    quality = result.get('voice_quality', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**🎯 Clarity:** {quality.get('clarity', 'N/A')}")
    with col2:
        st.info(f"**📊 Stability:** {quality.get('stability', 'N/A')}")
    with col3:
        st.info(f"**⚡ Energy:** {quality.get('energy_level', 'N/A')}")
    
    # Emotion distribution with better visualization
    st.markdown("### 🎭 Emotion Distribution")
    
    emotions_df = pd.DataFrame([
        {'Emotion': emotion.capitalize(), 'Confidence': score, 'Color': VOICE_EMOTION_MAPPING.get(emotion, {}).get('color', '#808080')}
        for emotion, score in result['all_emotions'].items()
    ]).sort_values('Confidence', ascending=False)
    
    fig = px.bar(
        emotions_df,
        x='Confidence',
        y='Emotion',
        orientation='h',
        color='Emotion',
        color_discrete_map={row['Emotion']: row['Color'] for _, row in emotions_df.iterrows()},
        title="Emotion Probabilities"
    )
    fig.update_layout(
        height=400,
        xaxis_title="Confidence",
        yaxis_title="Emotion",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Voice health indicators with severity
    if result['health_indicators']:
        st.markdown("### 🏥 Voice Health Assessment")
        
        for indicator in result['health_indicators']:
            color_map = {'high': '#EF4444', 'moderate': '#F59E0B', 'mild': '#3B82F6'}
            color = color_map.get(indicator['severity'], '#6B7280')
            
            with st.expander(f"⚠️ {indicator['issue']} - {indicator['severity'].upper()} Severity"):
                st.markdown(f"""
                <div style="padding: 1rem; background-color: {color}10; border-left: 5px solid {color}; border-radius: 5px;">
                    <p><strong>Clinical Term:</strong> {indicator.get('clinical_term', 'N/A')}</p>
                    <p><strong>Recommendation:</strong> {indicator['suggestion']}</p>
                    <p><strong>Exercises:</strong> {', '.join(indicator.get('exercises', ['Consult specialist']))}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Acoustic features visualization
    with st.expander("🔊 View Detailed Acoustic Features"):
        features = result['features']
        
        # Create tabs for different feature categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pitch", "Energy", "Spectral", "MFCC", "Rhythm"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Pitch Statistics**")
                st.markdown(f"- Mean: {features.get('pitch_mean', 0):.1f} Hz")
                st.markdown(f"- Std Dev: {features.get('pitch_std', 0):.1f} Hz")
                st.markdown(f"- Range: {features.get('pitch_range', 0):.1f} Hz")
                st.markdown(f"- Variability: {features.get('pitch_variability', 0):.3f}")
            with col2:
                st.markdown("**Pitch Range**")
                st.markdown(f"- Maximum: {features.get('pitch_max', 0):.1f} Hz")
                st.markdown(f"- Minimum: {features.get('pitch_min', 0):.1f} Hz")
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Energy Statistics**")
                st.markdown(f"- Mean: {features.get('energy_mean', 0):.4f}")
                st.markdown(f"- Std Dev: {features.get('energy_std', 0):.4f}")
                st.markdown(f"- Variability: {features.get('energy_variability', 0):.3f}")
            with col2:
                st.markdown("**Energy Range**")
                st.markdown(f"- Maximum: {features.get('energy_max', 0):.4f}")
                st.markdown(f"- Minimum: {features.get('energy_min', 0):.4f}")
        
        with tab3:
            st.markdown("**Spectral Features**")
            st.markdown(f"- Spectral Centroid: {features.get('spectral_centroid_mean', 0):.1f} Hz")
            st.markdown(f"- Spectral Bandwidth: {features.get('spectral_bandwidth_mean', 0):.1f} Hz")
            st.markdown(f"- Spectral Rolloff: {features.get('spectral_rolloff_mean', 0):.1f} Hz")
            st.markdown(f"- Zero Crossing Rate: {features.get('zcr_mean', 0):.4f}")
            st.markdown(f"- Harmonic Ratio: {features.get('harmonic_ratio', 0):.2f}")
        
        with tab4:
            st.markdown("**MFCC Features (First 12)**")
            mfcc_cols = st.columns(4)
            for i in range(12):
                with mfcc_cols[i % 4]:
                    st.markdown(f"MFCC {i+1}: {features.get(f'mfcc_mean_{i+1}', 0):.1f}")
        
        with tab5:
            st.markdown("**Rhythm Features**")
            st.markdown(f"- Tempo: {features.get('tempo', 0):.1f} BPM")
            st.markdown(f"- Speech Rate: {features.get('speech_rate', 0):.0f} spm")
            st.markdown(f"- Pause Count: {features.get('pause_count', 0)}")
            st.markdown(f"- Speech %: {features.get('speech_percentage', 0):.1f}%")
    
    # Recommended activities
    st.markdown("### 🌱 Recommended Activities")
    cols = st.columns(3)
    for i, activity in enumerate(emotion_data['activities']):
        with cols[i]:
            if st.button(f"✅ {activity}", key=f"voice_act_{i}"):
                st.success(f"Great! Try to {activity.lower()} now.")
                st.balloons()

# ==================== ENHANCED TEXT ANALYSIS FUNCTIONS ====================
def show_text_analysis_enhanced():
    """Enhanced text analysis UI"""
    st.markdown("## 📝 Enhanced Text Emotion Analysis")
    st.markdown("Analyze your text with advanced emotion detection and mental health insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter your text for comprehensive analysis:",
            height=250,
            placeholder="Share your thoughts, feelings, journal entries, or any text you'd like to analyze...",
            help="The more text you provide, the more accurate the analysis will be"
        )
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            analyze_btn = st.button("🔍 Analyze Text", use_container_width=True, type="primary")
        with col_b:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col_c:
            sample_btn = st.button("📋 Load Sample", use_container_width=True)
            
        if clear_btn:
            text_input = ""
            st.rerun()
        
        if sample_btn:
            text_input = """I've been feeling really anxious lately. Every time I think about the future, my heart races and I can't focus. 
            It's like there's this constant worry in the back of my mind. Sometimes I feel hopeless, but then I remember that 
            I have people who care about me. Today was a little better - I managed to go for a walk and it helped clear my mind. 
            I'm trying to stay positive and take things one day at a time."""
            st.rerun()
    
    with col2:
        st.markdown("### ⚙️ Analysis Settings")
        
        analysis_depth = st.radio(
            "Analysis Depth",
            ["Standard", "Deep", "Clinical"],
            help="Deep analysis provides more detailed insights"
        )
        
        context = st.selectbox(
            "Text Context",
            ["General", "Journal Entry", "Therapy Session", "Social Media", "Email", "Creative Writing", "Other"],
            help="Select the context for more accurate analysis"
        )
        
        st.markdown("### 📊 Text Stats Preview")
        if text_input:
            words = len(text_input.split())
            chars = len(text_input)
            st.info(f"Words: {words} | Characters: {chars}")
        else:
            st.info("Enter text to see preview")
        
        save_to_history = st.checkbox("Save to History", value=True)
    
    if analyze_btn and text_input:
        if len(text_input.split()) < 3:
            st.warning("Please enter at least 3 words for meaningful analysis.")
        else:
            with st.spinner("Performing comprehensive text analysis..."):
                # Perform enhanced analysis
                result = text_analyzer.analyze_text_comprehensive(text_input)
                result['context'] = context
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result['type'] = 'text_analysis'
                
                # Display results
                display_text_analysis_results_enhanced(result, analysis_depth)
                
                # Save to history
                if save_to_history:
                    st.session_state.analysis_history.append(result)
                    if save_analysis_to_history(st.session_state.username, result):
                        st.success("✅ Analysis saved to your history!")

def display_text_analysis_results_enhanced(result, depth):
    """Enhanced display with more details"""
    
    st.markdown("---")
    st.markdown("## 📊 Comprehensive Analysis Results")
    
    # Crisis alert (priority display)
    crisis_risk = result['crisis_risk']
    if crisis_risk['risk_level'] != 'none':
        risk_color = {
            'CRITICAL': '#7F1D1D',
            'HIGH': '#EF4444',
            'MODERATE': '#F59E0B',
            'LOW': '#FBBF24'
        }.get(crisis_risk['risk_level'], '#808080')
        
        risk_class = f"crisis-{crisis_risk['risk_level'].lower()}"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h3 style="color: {risk_color}; margin: 0;">⚠️ {crisis_risk['risk_level']} RISK DETECTED</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Severity Score: {crisis_risk['severity_score']}/10 | Indicators: {crisis_risk['indicator_count']}
            </p>
            <p style="margin: 0.5rem 0 0 0;">
                Your text contains indicators that may require immediate attention.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if crisis_risk['requires_immediate_attention']:
            st.error("🚨 **IMMEDIATE ACTION RECOMMENDED**")
            with st.expander("📞 Crisis Resources - Click for Help"):
                st.markdown("""
                **National Suicide Prevention Lifeline:** 988 or 1-800-273-8255  
                **Crisis Text Line:** Text HOME to 741741  
                **SAMHSA National Helpline:** 1-800-662-4357  
                **Emergency:** 911
                """)
    
    # Main emotion display
    wellness = result['wellness']
    dominant_emotion = result['dominant_emotion']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, {wellness['color']}20, {wellness['color']}40); 
                    border-radius: 15px; border-left: 5px solid {wellness['color']};">
            <h1 style="font-size: 3rem; margin: 0;">{wellness['icon']}</h1>
            <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {wellness['color']};">{dominant_emotion.title()}</p>
            <p style="margin: 0; font-size: 0.9rem;">Primary Emotion</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{result['dominant_score']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sentiment = result['sentiment']['combined']
        st.metric(
            "Sentiment Score", 
            f"{sentiment['compound']:.2f}",
            delta="Positive" if sentiment['compound'] > 0 else "Negative" if sentiment['compound'] < 0 else "Neutral"
        )
    
    with col3:
        stats = result['statistics']
        st.metric("Word Count", stats['word_count'])
    
    with col4:
        st.metric("Lexical Diversity", f"{stats['lexical_diversity']:.2f}")
    
    # Sentiment breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Sentiment Analysis")
        
        # Create sentiment gauge
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = sentiment['compound'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Sentiment"},
            gauge = {
                'axis': {'range': [-100, 100]},
                'bar': {'color': "#4A90E2"},
                'steps': [
                    {'range': [-100, -30], 'color': "#EF4444"},
                    {'range': [-30, 30], 'color': "#6B7280"},
                    {'range': [30, 100], 'color': "#22C55E"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment['compound'] * 100
                }
            }
        ))
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Sentiment Components")
        
        components_df = pd.DataFrame({
            'Component': ['Positive', 'Neutral', 'Negative'],
            'Score': [sentiment['positive'], sentiment['neutral'], sentiment['negative']],
            'Color': ['#22C55E', '#6B7280', '#EF4444']
        })
        
        fig = px.pie(
            components_df,
            values='Score',
            names='Component',
            color='Component',
            color_discrete_map={
                'Positive': '#22C55E',
                'Neutral': '#6B7280',
                'Negative': '#EF4444'
            }
        )
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Emotion distribution
    st.markdown("### 🎭 Emotion Distribution")
    
    emotions_df = pd.DataFrame([
        {'Emotion': emotion.capitalize(), 'Intensity': score * 100, 
         'Color': TEXT_EMOTION_MAPPING.get(emotion, {}).get('color', '#808080')}
        for emotion, score in result['emotions_detected'].items()
    ]).sort_values('Intensity', ascending=True)
    
    if not emotions_df.empty:
        fig = px.bar(
            emotions_df,
            x='Intensity',
            y='Emotion',
            orientation='h',
            color='Emotion',
            color_discrete_map={row['Emotion']: row['Color'] for _, row in emotions_df.iterrows()},
            title="Detected Emotions Intensity (%)"
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Deep analysis for Clinical depth
    if depth in ["Deep", "Clinical"]:
        st.markdown("### 🔍 Deep Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Phrases**")
            if result['key_phrases']:
                for phrase in result['key_phrases']:
                    st.markdown(f"- {phrase}")
            else:
                st.info("No key phrases detected")
        
        with col2:
            st.markdown("**Text Statistics**")
            stats = result['statistics']
            st.markdown(f"""
            - **Words:** {stats['word_count']} | **Unique:** {stats['unique_words']}
            - **Characters:** {stats['char_count']}
            - **Sentences:** {stats['sentence_count']}
            - **Avg Word Length:** {stats['avg_word_length']:.1f} chars
            - **Avg Sentence:** {stats['avg_sentence_length']:.1f} words
            """)
    
    if depth == "Clinical":
        st.markdown("### 🏥 Clinical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**POS Tagging Analysis**")
            st.markdown(f"""
            - **Nouns:** {stats['noun_count']}
            - **Verbs:** {stats['verb_count']}
            - **Adjectives:** {stats['adj_count']}
            - **Adverbs:** {stats['adv_count']}
            - **Pronouns:** {stats['pronoun_count']}
            """)
            
            # Calculate emotional tone
            if stats['adj_count'] + stats['adv_count'] > 0:
                emotional_ratio = (stats['adj_count'] + stats['adv_count']) / stats['word_count']
                st.markdown(f"**Emotional Density:** {emotional_ratio:.2f}")
        
        with col2:
            st.markdown("**Risk Assessment**")
            risk = result['crisis_risk']
            st.markdown(f"""
            - **Risk Level:** {risk['risk_level']}
            - **Severity Score:** {risk['severity_score']}/10
            - **Indicators Found:** {risk['indicator_count']}
            """)
            
            if risk['indicators']:
                st.markdown("**Specific Indicators:**")
                for indicator in risk['indicators'][:3]:
                    st.markdown(f"- {indicator}")
    
    # Wellness insights
    st.markdown("### 🌱 Wellness Insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="wellness-tip" style="border-left-color: {wellness['color']}; background-color: {wellness['color']}10;">
            <h4 style="color: {wellness['color']};">{wellness['status']}</h4>
            <p><strong>Recommendation:</strong> {wellness['suggestion']}</p>
            <p><strong>Wellness Tip:</strong> {wellness['wellness_tip']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Category:** {wellness['category']}")
        st.markdown("**Activities:**")
        for activity in wellness['activities']:
            st.markdown(f"- {activity}")
    
    # Full text view
    with st.expander("📝 View Full Text"):
        st.markdown(result['full_text'])

# ==================== FACIAL ANALYSIS FUNCTIONS ====================
def show_emotion_analysis():
    st.markdown("## 🎭 Facial Emotion Analysis")
    st.markdown("Analyze your facial expression to get mental health insights")
    
    if model is None:
        st.error("⚠️ Facial emotion recognition model not found. Please ensure 'emotiondetector.h5' exists in the project directory.")
        st.info("Text and voice analysis are still available in their respective tabs.")
        return
    
    option = st.radio("Choose Input Method", ["📷 Live Camera", "📁 Upload Image"], horizontal=True)
    
    source = None
    if option == "📷 Live Camera":
        source = st.camera_input("Take a snapshot")
    else:
        source = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    def process_and_predict(input_source, is_upload=False):
        file_bytes = np.asarray(bytearray(input_source.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if is_upload:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
        else:
            if is_upload:
                st.info("No face bounding box detected. Processing the full image.")
                roi_gray = gray
            else:
                return None, None, "No face detected. Please ensure good lighting and positioning."
        
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized / 255.0
        final_input = np.reshape(roi_normalized, (1, 48, 48, 1))
        
        preds = model.predict(final_input, verbose=0)[0]
        return roi_resized, preds, None
    
    if source:
        is_upload_mode = (option == "📁 Upload Image")
        
        with st.spinner("Analyzing..."):
            cropped_face, predictions, error = process_and_predict(source, is_upload=is_upload_mode)
        
        if error:
            st.error(error)
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(cropped_face, caption="Analyzed Face", width=200)
                
                max_conf = np.max(predictions)
                st.markdown(f"### Confidence: {max_conf*100:.1f}%")
                st.progress(min(int(max_conf*100), 100))
            
            with col2:
                dominant_emotion = CLASS_NAMES[np.argmax(predictions)]
                mental_health = MENTAL_HEALTH_MAPPING[dominant_emotion]
                
                st.markdown(f"""
                <div class="wellness-tip">
                    <h3 style='color: {mental_health["color"]};'>Primary Emotion: {dominant_emotion.upper()}</h3>
                    <p><strong>Status:</strong> {mental_health['status']}</p>
                    <p><strong>Wellness Insight:</strong> {mental_health['wellness_tip']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[c.capitalize() for c in CLASS_NAMES],
                        y=predictions * 100,
                        marker_color=[MENTAL_HEALTH_MAPPING[c]['color'] for c in CLASS_NAMES],
                        text=[f"{p*100:.1f}%" for p in predictions],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Emotion Distribution",
                    xaxis_title="Emotion",
                    yaxis_title="Confidence (%)",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🌱 Recommended Activities")
                cols = st.columns(3)
                for i, activity in enumerate(mental_health['activities']):
                    with cols[i]:
                        if st.button(f"✅ {activity}", key=f"activity_{i}"):
                            st.success(f"Great! Try to {activity.lower()} now.")
                
                if st.button("💾 Save to History", use_container_width=True):
                    analysis_data = {
                        'type': 'facial_analysis',
                        'dominant_emotion': dominant_emotion,
                        'confidence': float(max_conf),
                        'all_emotions': {emotion: float(conf) for emotion, conf in zip(CLASS_NAMES, predictions)},
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.analysis_history.append(analysis_data)
                    if save_analysis_to_history(st.session_state.username, analysis_data):
                        st.success("✅ Analysis saved to your history!")
                    else:
                        st.error("Failed to save to history")

# ==================== HISTORY DISPLAY ====================
def show_history():
    st.markdown("## 📊 Your Analysis History")
    
    if not st.session_state.analysis_history:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <img src="https://img.icons8.com/fluency/240/000000/sad.png" 
                     style="width: 120px; height: auto; opacity: 0.5;">
                <h3 style="color: #666;">No Analysis History Yet</h3>
                <p>Start by analyzing your emotions using the Facial Analysis, Text Analysis, or Voice Analysis page!</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    analysis_type = st.selectbox(
        "Filter by Analysis Type",
        ["All", "Facial Analysis", "Text Analysis", "Voice Analysis"]
    )
    
    filtered_history = []
    for entry in st.session_state.analysis_history:
        entry_type = entry.get('type', 'facial_analysis')
        if analysis_type == "All":
            filtered_history.append(entry)
        elif analysis_type == "Facial Analysis" and entry_type == 'facial_analysis':
            filtered_history.append(entry)
        elif analysis_type == "Text Analysis" and entry_type == 'text_analysis':
            filtered_history.append(entry)
        elif analysis_type == "Voice Analysis" and entry_type == 'voice_analysis':
            filtered_history.append(entry)
    
    for i, entry in enumerate(reversed(filtered_history)):
        entry_type = entry.get('type', 'facial_analysis')
        
        if entry_type == 'text_analysis':
            wellness = entry.get('wellness', {})
            color = wellness.get('color', '#808080')
            
            with st.expander(f"📝 Text Analysis {len(filtered_history)-i}: {entry['timestamp']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Text Preview:** {entry.get('text_preview', 'N/A')}")
                    st.markdown(f"**Primary Emotion:** <span style='color:{color}'>{entry['dominant_emotion'].upper()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Sentiment Score:** {entry['sentiment']['combined']['compound']:.2f}")
                    st.markdown(f"**Word Count:** {entry['statistics']['word_count']}")
                    st.markdown(f"**Context:** {entry.get('context', 'General')}")
                
                with col2:
                    if st.button(f"View Details", key=f"view_text_{i}"):
                        st.session_state.current_text_analysis = entry
                        st.rerun()
                    
                    if st.button(f"Delete", key=f"del_text_{i}"):
                        st.session_state.analysis_history.remove(entry)
                        st.rerun()
        
        elif entry_type == 'voice_analysis':
            emotion_data = entry.get('emotion_data', VOICE_EMOTION_MAPPING['neutral'])
            color = emotion_data.get('color', '#808080')
            
            with st.expander(f"🎤 Voice Analysis {len(filtered_history)-i}: {entry['timestamp']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Emotion:** <span style='color:{color}'>{entry['dominant_emotion'].upper()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {entry['confidence']*100:.1f}%")
                    st.markdown(f"**Duration:** {entry.get('duration', 0):.1f}s")
                    st.markdown(f"**Avg Pitch:** {entry.get('features', {}).get('pitch_mean', 0):.0f} Hz")
                
                with col2:
                    if st.button(f"View Details", key=f"view_voice_{i}"):
                        st.session_state.current_voice_analysis = entry
                        st.rerun()
                    
                    if st.button(f"Delete", key=f"del_voice_{i}"):
                        st.session_state.analysis_history.remove(entry)
                        st.rerun()
        
        else:  # facial analysis
            emotion = entry['dominant_emotion']
            color = MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#808080')
            
            with st.expander(f"😊 Facial Analysis {len(filtered_history)-i}: {entry['timestamp']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Emotion:** <span style='color:{color}'>{emotion.upper()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {entry.get('confidence', 0)*100:.1f}%")
                
                with col2:
                    if st.button(f"View Details", key=f"view_face_{i}"):
                        st.session_state.current_analysis = entry
                        st.rerun()
                    
                    if st.button(f"Delete", key=f"del_face_{i}"):
                        st.session_state.analysis_history.remove(entry)
                        st.rerun()

# ==================== ANALYTICS DISPLAY WITH REPORT GENERATION ====================
def show_analytics():
    st.markdown("## 📈 Advanced Analytics")
    
    if not st.session_state.analysis_history:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <img src="https://img.icons8.com/fluency/240/000000/sad.png" 
                     style="width: 120px; height: auto; opacity: 0.5;">
                <h3 style="color: #666;">No Data Available</h3>
                <p style="color: #666;">Complete some emotion analyses first to see your analytics!</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns(2)
    with col1:
        analysis_type_filter = st.selectbox(
            "Analysis Type",
            ["All", "Facial Only", "Text Only", "Voice Only", "Combined"]
        )
    with col2:
        date_range = st.selectbox(
            "Date Range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
        )
    
    # Filter data based on analysis type
    filtered_data = []
    for entry in st.session_state.analysis_history:
        entry_type = entry.get('type', 'facial_analysis')
        
        if analysis_type_filter == "All":
            filtered_data.append(entry)
        elif analysis_type_filter == "Facial Only" and entry_type == 'facial_analysis':
            filtered_data.append(entry)
        elif analysis_type_filter == "Text Only" and entry_type == 'text_analysis':
            filtered_data.append(entry)
        elif analysis_type_filter == "Voice Only" and entry_type == 'voice_analysis':
            filtered_data.append(entry)
        elif analysis_type_filter == "Combined":
            filtered_data.append(entry)
    
    # Apply date filter with error handling
    start_date = None
    end_date = datetime.now()
    
    if date_range != "All time":
        try:
            days = int(date_range.split()[1])
            cutoff = datetime.now() - timedelta(days=days)
            start_date = cutoff
            
            date_filtered = []
            for e in filtered_data:
                if 'timestamp' in e:
                    try:
                        date_str = e['timestamp'][:10]
                        entry_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if entry_date >= cutoff:
                            date_filtered.append(e)
                    except (ValueError, KeyError, TypeError):
                        date_filtered.append(e)
                else:
                    date_filtered.append(e)
            
            filtered_data = date_filtered
        except Exception as e:
            st.warning(f"Date filtering issue: {str(e)}")
    else:
        try:
            dates = []
            for e in filtered_data:
                if 'timestamp' in e:
                    try:
                        date_str = e['timestamp'][:10]
                        dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
                    except:
                        pass
            if dates:
                start_date = min(dates)
        except:
            start_date = datetime.now() - timedelta(days=30)
    
    if not filtered_data:
        st.warning("No data available for the selected filters.")
        return
    
    # Separate data by type with safe access
    facial_data = []
    text_data = []
    voice_data = []
    
    for entry in filtered_data:
        entry_type = entry.get('type', 'facial_analysis')
        if entry_type == 'text_analysis':
            text_data.append(entry)
        elif entry_type == 'voice_analysis':
            voice_data.append(entry)
        else:
            facial_data.append(entry)
    
    # Create DataFrames safely
    facial_df = pd.DataFrame(facial_data) if facial_data else pd.DataFrame()
    text_df = pd.DataFrame(text_data) if text_data else pd.DataFrame()
    voice_df = pd.DataFrame(voice_data) if voice_data else pd.DataFrame()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyses", len(filtered_data))
    with col2:
        st.metric("Facial Analyses", len(facial_data))
    with col3:
        st.metric("Text Analyses", len(text_data))
    with col4:
        st.metric("Voice Analyses", len(voice_data))
    
    # Comparison Charts
    if facial_data or text_data or voice_data:
        st.markdown("### 📊 Analysis Type Comparison")
        
        type_counts = {
            'Facial': len(facial_data),
            'Text': len(text_data),
            'Voice': len(voice_data)
        }
        type_counts = {k: v for k, v in type_counts.items() if v > 0}
        
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Analysis Type Distribution",
            color_discrete_sequence=['#4A90E2', '#9C27B0', '#FF6B6B']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Voice Analysis Charts
    if voice_data:
        st.markdown("### 🎤 Voice Analysis Overview")
        
        avg_pitch = np.mean([e.get('features', {}).get('pitch_mean', 0) for e in voice_data])
        avg_energy = np.mean([e.get('features', {}).get('energy_mean', 0) for e in voice_data])
        avg_confidence = np.mean([e.get('confidence', 0) for e in voice_data]) * 100
        avg_quality = np.mean([e.get('quality_score', 0) for e in voice_data]) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Pitch", f"{avg_pitch:.0f} Hz")
        with col2:
            st.metric("Avg Energy", f"{avg_energy:.3f}")
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        with col4:
            st.metric("Avg Quality", f"{avg_quality:.1f}%")
        
        voice_emotions = [e['dominant_emotion'] for e in voice_data]
        emotion_counts = Counter(voice_emotions)
        
        fig_voice = px.bar(
            x=[VOICE_EMOTION_MAPPING.get(e, {}).get('status', e.capitalize()) for e in emotion_counts.keys()],
            y=list(emotion_counts.values()),
            title="Voice Emotion Distribution",
            color=[VOICE_EMOTION_MAPPING.get(e, {}).get('color', '#808080') for e in emotion_counts.keys()],
            labels={'x': 'Emotion', 'y': 'Count'}
        )
        st.plotly_chart(fig_voice, use_container_width=True)
        
        health_alerts = 0
        for entry in voice_data:
            if 'health_indicators' in entry and entry['health_indicators']:
                health_alerts += len(entry['health_indicators'])
        
        if health_alerts > 0:
            st.warning(f"⚠️ {health_alerts} voice health alerts detected in your history")
    
    # Facial Analysis Charts
    if facial_data:
        st.markdown("### 🎭 Facial Analysis Timeline")
        
        facial_with_time = []
        for entry in facial_data:
            if 'timestamp' in entry:
                facial_with_time.append(entry)
        
        if facial_with_time:
            facial_timeline_df = pd.DataFrame(facial_with_time)
            fig_timeline = create_emotion_timeline(facial_timeline_df)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("### 📈 Facial Analysis Radar")
        if facial_with_time:
            facial_radar_df = pd.DataFrame(facial_with_time)
            fig_radar = create_emotion_radar(facial_radar_df)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("### 📅 Facial Weekly Pattern")
        if facial_with_time:
            facial_weekly_df = pd.DataFrame(facial_with_time)
            fig_weekly = create_weekly_pattern(facial_weekly_df)
            if fig_weekly:
                st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Text Analysis Overview
    if text_data:
        st.markdown("### 📝 Text Analysis Overview")
        
        avg_sentiment = 0
        avg_word_count = 0
        crisis_count = 0
        
        for entry in text_data:
            if 'sentiment' in entry and isinstance(entry['sentiment'], dict):
                if 'combined' in entry['sentiment']:
                    avg_sentiment += entry['sentiment']['combined'].get('compound', 0)
            
            if 'statistics' in entry and isinstance(entry['statistics'], dict):
                avg_word_count += entry['statistics'].get('word_count', 0)
            
            if 'crisis_risk' in entry and isinstance(entry['crisis_risk'], dict):
                if entry['crisis_risk'].get('risk_level', 'none') != 'none':
                    crisis_count += 1
        
        if text_data:
            avg_sentiment /= len(text_data)
            avg_word_count /= len(text_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        with col2:
            st.metric("Avg Word Count", f"{avg_word_count:.0f}")
        with col3:
            st.metric("Crisis Alerts", crisis_count)
        
        text_emotions = []
        for entry in text_data:
            if 'dominant_emotion' in entry:
                text_emotions.append(entry['dominant_emotion'])
        
        if text_emotions:
            emotion_counts = Counter(text_emotions)
            fig_text_dist = px.bar(
                x=[e.capitalize() for e in emotion_counts.keys()],
                y=list(emotion_counts.values()),
                title="Text Emotion Distribution",
                color=[TEXT_EMOTION_MAPPING.get(e, {}).get('color', '#808080') for e in emotion_counts.keys()],
                labels={'x': 'Emotion', 'y': 'Count'}
            )
            st.plotly_chart(fig_text_dist, use_container_width=True)
    
    # PDF Report Generation Section
    st.markdown("---")
    st.markdown("### 📋 Clinical Report Generation")
    st.markdown("Generate a comprehensive clinical report based on your analytics data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_title = st.text_input(
            "Report Title", 
            value=f"NeuroWell Clinical Assessment - {st.session_state.username}",
            key="report_title"
        )
    
    with col2:
        report_type = st.selectbox(
            "Report Type",
            ["Comprehensive Report", "Facial Analysis Only", "Text Analysis Only", "Voice Analysis Only", "Summary Report"],
            key="report_type"
        )
    
    with col3:
        include_charts = st.checkbox("Include Charts in Report", value=True, key="include_charts")
    
    report_data = []
    if report_type == "Facial Analysis Only":
        report_data = facial_data
        report_df = facial_df.copy() if not facial_df.empty else pd.DataFrame()
    elif report_type == "Text Analysis Only":
        report_data = text_data
        report_df = pd.DataFrame()
    elif report_type == "Voice Analysis Only":
        report_data = voice_data
        report_df = pd.DataFrame()
    else:
        report_data = filtered_data
        report_df = facial_df.copy() if not facial_df.empty else pd.DataFrame()
    
    if st.button("📄 Generate Hospital-Grade PDF Report", use_container_width=True, type="primary"):
        if not report_data:
            st.error("No data available for the selected report type!")
        else:
            with st.spinner("Generating professional clinical report..."):
                try:
                    start_date_str = start_date.strftime("%Y-%m-%d") if start_date else "N/A"
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    
                    final_title = f"{report_title} - {report_type}"
                    
                    if report_type == "Facial Analysis Only":
                        final_title += " (Facial Data)"
                    elif report_type == "Text Analysis Only":
                        final_title += " (Text Data)"
                    elif report_type == "Voice Analysis Only":
                        final_title += " (Voice Data)"
                    
                    pdf_bytes = None
                    
                    if report_type == "Text Analysis Only" and text_data:
                        pdf_bytes = generate_text_pdf_report(
                            text_data,
                            st.session_state.username,
                            start_date_str,
                            end_date_str,
                            final_title
                        )
                    elif report_type == "Voice Analysis Only" and voice_data:
                        pdf_bytes = generate_voice_pdf_report(
                            voice_data,
                            st.session_state.username,
                            start_date_str,
                            end_date_str,
                            final_title
                        )
                    elif not report_df.empty:
                        pdf_bytes = generate_professional_pdf_report(
                            report_df, 
                            st.session_state.username,
                            start_date_str,
                            end_date_str,
                            final_title
                        )
                    else:
                        st.warning("No valid data available for PDF generation.")
                    
                    if pdf_bytes:
                        filename = f"NeuroWell_Report_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        
                        st.download_button(
                            label="📥 Download Clinical Report (PDF)",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            key="download_report"
                        )
                        st.success("✅ Clinical PDF report generated successfully!")
                        st.balloons()
                    else:
                        if text_data and report_type != "Text Analysis Only":
                            st.info("📝 Text analysis data detected. Try selecting 'Text Analysis Only' report type.")
                        if facial_data and report_type != "Facial Analysis Only":
                            st.info("🎭 Facial analysis data detected. Try selecting 'Facial Analysis Only' report type.")
                        if voice_data and report_type != "Voice Analysis Only":
                            st.info("🎤 Voice analysis data detected. Try selecting 'Voice Analysis Only' report type.")
                
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.exception(e)
    
    st.markdown("### 📊 Additional Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Data as CSV", use_container_width=True):
            export_data = []
            for entry in filtered_data:
                entry_type = entry.get('type', 'facial_analysis')
                
                if entry_type == 'text_analysis':
                    export_data.append({
                        'Date': entry.get('timestamp', 'N/A')[:10],
                        'Time': entry.get('timestamp', 'N/A')[11:16] if len(entry.get('timestamp', '')) > 16 else 'N/A',
                        'Type': 'Text Analysis',
                        'Primary Emotion': entry.get('dominant_emotion', 'N/A'),
                        'Sentiment': f"{entry.get('sentiment', {}).get('combined', {}).get('compound', 0):.2f}",
                        'Word Count': entry.get('statistics', {}).get('word_count', 0) if isinstance(entry.get('statistics'), dict) else 0,
                        'Crisis Risk': entry.get('crisis_risk', {}).get('risk_level', 'none') if isinstance(entry.get('crisis_risk'), dict) else 'none'
                    })
                elif entry_type == 'voice_analysis':
                    export_data.append({
                        'Date': entry.get('timestamp', 'N/A')[:10],
                        'Time': entry.get('timestamp', 'N/A')[11:16] if len(entry.get('timestamp', '')) > 16 else 'N/A',
                        'Type': 'Voice Analysis',
                        'Primary Emotion': entry.get('dominant_emotion', 'N/A'),
                        'Confidence': f"{entry.get('confidence', 0)*100:.1f}%",
                        'Pitch (Hz)': entry.get('features', {}).get('pitch_mean', 0),
                        'Health Alerts': len(entry.get('health_indicators', []))
                    })
                else:
                    export_data.append({
                        'Date': entry.get('timestamp', 'N/A')[:10],
                        'Time': entry.get('timestamp', 'N/A')[11:16] if len(entry.get('timestamp', '')) > 16 else 'N/A',
                        'Type': 'Facial Analysis',
                        'Primary Emotion': entry.get('dominant_emotion', 'N/A'),
                        'Confidence': f"{entry.get('confidence', 0)*100:.1f}%",
                        'Word Count': 'N/A',
                        'Crisis Risk': 'N/A'
                    })
            
            if export_data:
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"neurowell_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📥 Export Data as JSON", use_container_width=True):
            json_str = json.dumps(filtered_data, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"neurowell_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# ==================== PROFILE DISPLAY ====================
def show_profile():
    st.markdown("## 👤 Your Profile")
    
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        if st.session_state.username in users:
            user_data = users[st.session_state.username]
            
            st.markdown("### 📸 Profile Picture")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                profile_image_path = st.session_state.profile_image or get_profile_image(st.session_state.username)
                if profile_image_path and os.path.exists(profile_image_path):
                    with open(profile_image_path, "rb") as f:
                        img_data = f.read()
                        img_base64 = base64.b64encode(img_data).decode()
                    st.markdown(f"""
                    <div class="profile-image-container">
                        <img src="data:image/png;base64,{img_base64}" class="profile-image">
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="profile-image-container">
                        <img src="https://img.icons8.com/fluency/240/000000/user-male-circle.png" class="profile-image">
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Upload New Profile Picture")
                uploaded_file = st.file_uploader(
                    "Choose an image...", 
                    type=["jpg", "jpeg", "png"],
                    key="profile_uploader"
                )
                
                if uploaded_file is not None:
                    st.image(uploaded_file, caption="Preview", width=150)
                    
                    if st.button("💾 Save Profile Picture", use_container_width=True):
                        image_data = uploaded_file.getvalue()
                        success, result = save_profile_image(st.session_state.username, image_data)
                        
                        if success:
                            st.session_state.profile_image = result
                            st.success("✅ Profile picture updated successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error saving image: {result}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Account Information")
                st.info(f"**Username:** {st.session_state.username}")
                st.info(f"**Email:** {user_data['email']}")
                st.info(f"**Member since:** {user_data['created_at']}")
                st.info(f"**Total analyses:** {len(user_data.get('history', []))}")
            
            with col2:
                st.markdown("### Settings")
                settings = user_data.get('settings', {
                    'notifications': True,
                    'save_history': True,
                    'theme': 'Light'
                })
                
                notifications = st.toggle("Email notifications", value=settings.get('notifications', True))
                save_history = st.toggle("Save analysis history", value=settings.get('save_history', True))
                theme = st.selectbox(
                    "Theme", 
                    ["Light", "Dark", "System default"], 
                    index=["Light", "Dark", "System default"].index(settings.get('theme', 'Light'))
                )
                
                if st.button("💾 Save Settings", use_container_width=True):
                    new_settings = {
                        'notifications': notifications,
                        'save_history': save_history,
                        'theme': theme
                    }
                    if update_user_settings(st.session_state.username, new_settings):
                        st.success("Settings saved successfully!")
                    else:
                        st.error("Failed to save settings")
            
            st.markdown("---")
            
            st.markdown("### Data Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export My Data", use_container_width=True):
                    export_data = {
                        'username': st.session_state.username,
                        'email': user_data['email'],
                        'created_at': user_data['created_at'],
                        'settings': user_data.get('settings', {}),
                        'analysis_history': user_data.get('history', [])
                    }
                    
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"neurowell_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                    if st.session_state.analysis_history:
                        st.session_state.analysis_history = []
                        users[st.session_state.username]['history'] = []
                        with open(USER_DB_FILE, 'w') as f:
                            json.dump(users, f, indent=4)
                        st.success("History cleared successfully!")
                        st.rerun()
            
            st.markdown("---")
            
            st.markdown("### ⚠️ Danger Zone")
            with st.expander("Delete Account"):
                st.warning("This action cannot be undone. All your data will be permanently deleted.")
                confirm = st.text_input("Type 'DELETE' to confirm")
                if st.button("🗑️ Permanently Delete Account", use_container_width=True, type="primary"):
                    if confirm == "DELETE":
                        del users[st.session_state.username]
                        with open(USER_DB_FILE, 'w') as f:
                            json.dump(users, f, indent=4)
                        
                        profile_path = get_profile_image(st.session_state.username)
                        if profile_path and os.path.exists(profile_path):
                            os.remove(profile_path)
                        
                        st.session_state.logged_in = False
                        st.session_state.username = None
                        st.session_state.analysis_history = []
                        st.session_state.profile_image = None
                        st.success("Account deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Please type 'DELETE' to confirm")
    
    except Exception as e:
        st.error(f"Error loading profile: {e}")

# ==================== HOME PAGE ====================
def show_home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            {get_logo_html('medium')}
            <h1 style="color: #4A90E2; margin: 0;">Welcome to NeuroWell</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        Your comprehensive mental health companion that uses facial emotion recognition, 
        text analysis, and voice analysis to help you understand and manage your emotional well-being.
        
        ### 🌟 Key Features:
        - **Facial Emotion Analysis**: Understand your emotional state from facial expressions
        - **Enhanced Text Analysis**: Advanced NLP with emotion detection and crisis assessment
        - **Enhanced Voice Analysis**: ML-based emotion detection with 20+ acoustic features
        - **Mental Health Insights**: Personalized suggestions based on your emotions
        - **Track Your Journey**: Monitor emotional patterns over time
        - **Advanced Analytics**: Visualize trends with multiple chart types
        - **Voice Health Monitoring**: Get alerts about potential voice strain
        - **Clinical Reports**: Generate hospital-grade PDF reports
        
        ### 📊 Quick Start:
        """)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🎭 Try Facial Analysis", use_container_width=True):
                st.session_state.current_page = "Facial Analysis"
                st.rerun()
        
        with col_b:
            if st.button("📝 Try Enhanced Text", use_container_width=True):
                st.session_state.current_page = "Text Analysis"
                st.rerun()
        
        with col_c:
            if st.button("🎤 Try Enhanced Voice", use_container_width=True):
                st.session_state.current_page = "Voice Analysis"
                st.rerun()
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; animation: float 3s ease-in-out infinite;">
            {get_logo_html('xlarge')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌱 Today's Tip")
        st.info("Your voice, face, and words all tell a story. Our enhanced AI now understands them better than ever.")
        
        if st.session_state.analysis_history:
            st.markdown("### 📊 Your Stats")
            total = len(st.session_state.analysis_history)
            facial_count = len([e for e in st.session_state.analysis_history if e.get('type') == 'facial_analysis'])
            text_count = len([e for e in st.session_state.analysis_history if e.get('type') == 'text_analysis'])
            voice_count = len([e for e in st.session_state.analysis_history if e.get('type') == 'voice_analysis'])
            
            st.metric("Total Analyses", total)
            st.metric("Facial Analyses", facial_count)
            st.metric("Text Analyses", text_count)
            st.metric("Voice Analyses", voice_count)

# ==================== MENTAL HEALTH TIPS ====================
def show_mental_health_tips():
    st.markdown("## 💚 Mental Health Tips")
    st.markdown("Personalized wellness tips based on your emotional state")
    
    tab1, tab2, tab3 = st.tabs(["🎭 Facial Emotions", "📝 Text Emotions", "🎤 Voice Emotions"])
    
    with tab1:
        selected_emotion = st.selectbox(
            "Filter tips by emotion",
            ["All"] + [e.capitalize() for e in CLASS_NAMES],
            key="facial_tips"
        )
        
        if selected_emotion == "All":
            emotions_to_show = CLASS_NAMES
        else:
            emotions_to_show = [selected_emotion.lower()]
        
        for emotion in emotions_to_show:
            data = MENTAL_HEALTH_MAPPING[emotion]
            with st.expander(f"{data['status']} - {emotion.capitalize()}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    **🧠 Wellness Insight**  
                    {data['wellness_tip']}
                    
                    **💡 Quick Tip**  
                    {data['suggestion']}
                    
                    **🎯 Suggested Activities**  
                    • {data['activities'][0]}  
                    • {data['activities'][1]}  
                    • {data['activities'][2]}
                    """)
                with col2:
                    st.markdown(f"<div style='background-color:{data['color']}20; padding:1rem; border-radius:10px;'>", unsafe_allow_html=True)
                    st.markdown(f"### When you feel {emotion}")
                    st.markdown("Remember: This emotion will pass. Be kind to yourself.")
                    st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        selected_text_emotion = st.selectbox(
            "Filter tips by emotion",
            ["All"] + [e.capitalize() for e in TEXT_EMOTION_MAPPING.keys()],
            key="text_tips"
        )
        
        if selected_text_emotion == "All":
            text_emotions_to_show = TEXT_EMOTION_MAPPING.keys()
        else:
            text_emotions_to_show = [selected_text_emotion.lower()]
        
        for emotion in text_emotions_to_show:
            data = TEXT_EMOTION_MAPPING[emotion]
            with st.expander(f"{data['status']} - {emotion.capitalize()}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    **🧠 Wellness Insight**  
                    {data['wellness_tip']}
                    
                    **💡 Quick Tip**  
                    {data['suggestion']}
                    
                    **🎯 Suggested Activities**  
                    • {data['activities'][0]}  
                    • {data['activities'][1]}  
                    • {data['activities'][2]}
                    """)
                with col2:
                    st.markdown(f"<div style='background-color:{data['color']}20; padding:1rem; border-radius:10px;'>", unsafe_allow_html=True)
                    st.markdown(f"### When you feel {emotion} in your writing")
                    st.markdown("Your words reflect your inner state. Acknowledge and honor your feelings.")
                    st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        selected_voice_emotion = st.selectbox(
            "Filter tips by emotion",
            ["All"] + [e.capitalize() for e in VOICE_EMOTION_MAPPING.keys()],
            key="voice_tips"
        )
        
        if selected_voice_emotion == "All":
            voice_emotions_to_show = VOICE_EMOTION_MAPPING.keys()
        else:
            voice_emotions_to_show = [selected_voice_emotion.lower()]
        
        for emotion in voice_emotions_to_show:
            data = VOICE_EMOTION_MAPPING[emotion]
            with st.expander(f"{data['status']} - {emotion.capitalize()}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    **🧠 Wellness Insight**  
                    {data['wellness_tip']}
                    
                    **💡 Quick Tip**  
                    {data['suggestion']}
                    
                    **🎯 Suggested Activities**  
                    • {data['activities'][0]}  
                    • {data['activities'][1]}  
                    • {data['activities'][2]}
                    
                    **🎵 Voice Care**  
                    • Stay hydrated  
                    • Avoid shouting  
                    • Practice deep breathing
                    """)
                with col2:
                    st.markdown(f"<div style='background-color:{data['color']}20; padding:1rem; border-radius:10px;'>", unsafe_allow_html=True)
                    st.markdown(f"### When your voice shows {emotion}")
                    st.markdown("Your voice is a powerful indicator of your emotional state.")
                    st.markdown("</div>", unsafe_allow_html=True)

# ==================== MAIN APP ====================
def show_main_app():
    with st.sidebar:
        profile_image_path = st.session_state.profile_image or get_profile_image(st.session_state.username)
        
        if profile_image_path and os.path.exists(profile_image_path):
            with open(profile_image_path, "rb") as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode()
            profile_html = f'<img src="data:image/png;base64,{img_base64}" class="sidebar-profile-image">'
        else:
            profile_html = '<img src="https://img.icons8.com/fluency/96/000000/user-male-circle.png" class="sidebar-profile-image">'
        
        st.markdown(f"""
        <div class="sidebar-profile">
            {profile_html}
            <h3 style="color: #4A90E2; margin: 0.5rem 0 0 0;">{st.session_state.username}</h3>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">NeuroWell User</p>
        </div>
        
        <div style="text-align: center; margin-bottom: 1rem;">
            {get_logo_html('sidebar')}
            <h2 style="color: #4A90E2; margin: 0.5rem 0 0 0; 
                       background: linear-gradient(45deg, #4A90E2, #9C27B0);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                NeuroWell
            </h2>
            <p style="color: #666; font-size: 0.8rem;">Enhanced AI Edition</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Facial Analysis", "Text Analysis", "Voice Analysis", "Mental Health Tips", "History", "Analytics", "Profile"],
            icons=["house", "camera", "pencil-square", "mic", "heart", "clock-history", "graph-up", "person"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#4A90E2", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#4A90E2"},
            }
        )
        
        st.markdown("---")
        
        if st.session_state.analysis_history:
            total = len(st.session_state.analysis_history)
            st.markdown(f"### 📊 Quick Stats")
            st.markdown(f"Total Analyses: **{total}**")
            
            if st.session_state.analysis_history:
                last = st.session_state.analysis_history[-1].get('timestamp', 'N/A')
                st.markdown(f"Last: **{last[:10]}**")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.analysis_history = []
            st.session_state.profile_image = None
            st.rerun()
    
    if selected == "Home":
        show_home_page()
    elif selected == "Facial Analysis":
        show_emotion_analysis()
    elif selected == "Text Analysis":
        show_text_analysis_enhanced()
    elif selected == "Voice Analysis":
        show_voice_analysis_enhanced()
    elif selected == "Mental Health Tips":
        show_mental_health_tips()
    elif selected == "History":
        show_history()
    elif selected == "Analytics":
        show_analytics()
    elif selected == "Profile":
        show_profile()

# ==================== MAIN EXECUTION ====================
def main():
    if not st.session_state.logged_in:
        show_login_signup()
    else:
        show_main_app()

if __name__ == "__main__": 
    main()