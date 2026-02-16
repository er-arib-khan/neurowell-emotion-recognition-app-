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

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
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

# ==================== TEXT ANALYSIS CONFIGURATION ====================
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

# ==================== TEXT ANALYZER CLASS ====================
class TextEmotionAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_patterns = self.load_emotion_patterns()
        
    def load_emotion_patterns(self):
        """Load emotion patterns for text analysis"""
        return {
            'joy': {
                'words': ['happy', 'joy', 'delighted', 'pleased', 'grateful', 'thankful', 
                         'hopeful', 'optimistic', 'positive', 'good', 'great', 'excellent',
                         'wonderful', 'fantastic', 'amazing', 'beautiful', 'love', 'loving',
                         'glad', 'cheerful', 'excited', 'thrilled', 'blessed'],
                'phrases': ['i am happy', 'feeling good', 'so glad', 'very happy']
            },
            'sadness': {
                'words': ['sad', 'depressed', 'unhappy', 'miserable', 'gloomy', 'hopeless',
                         'devastated', 'heartbroken', 'grief', 'mourning', 'distressed',
                         'tearful', 'crying', 'sorrow', 'despair', 'lonely', 'alone',
                         'hurt', 'pain', 'suffering', 'blue', 'down'],
                'phrases': ['feel sad', 'so sad', 'very sad', 'i am sad']
            },
            'anger': {
                'words': ['angry', 'mad', 'furious', 'frustrated', 'irritated', 'annoyed',
                         'hostile', 'aggressive', 'outraged', 'resentful', 'bitter',
                         'frustration', 'irritation', 'hate', 'hatred', 'rage', 'fury'],
                'phrases': ['i am angry', 'so angry', 'very angry', 'makes me mad']
            },
            'fear': {
                'words': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous',
                         'panicked', 'frightened', 'alarmed', 'dread', 'fearful',
                         'anxiety', 'panic', 'phobia', 'stress', 'stressed'],
                'phrases': ['i am afraid', 'so scared', 'very anxious', 'worried about']
            },
            'surprise': {
                'words': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned',
                         'unexpected', 'unanticipated', 'sudden', 'startled',
                         'remarkable', 'extraordinary', 'wow', 'oh', 'ah'],
                'phrases': ['i am surprised', 'so surprised', 'can\'t believe']
            },
            'trust': {
                'words': ['trust', 'confident', 'believe', 'faith', 'reliable', 'dependable',
                         'assured', 'certain', 'sure', 'confidence', 'trusting',
                         'trustworthy', 'honest', 'sincere', 'loyal'],
                'phrases': ['i trust', 'i believe', 'i am confident']
            },
            'anticipation': {
                'words': ['expect', 'anticipate', 'await', 'look forward', 'prospective',
                         'pending', 'upcoming', 'future', 'expecting', 'awaiting',
                         'preparing', 'hope', 'hoping', 'wish', 'wishing'],
                'phrases': ['looking forward', 'can\'t wait', 'hope for']
            }
        }
    
    def analyze_sentiment_vader(self, text):
        """VADER sentiment analysis"""
        return self.sia.polarity_scores(text)
    
    def analyze_sentiment_textblob(self, text):
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def detect_emotions(self, text):
        """Detect emotions in text"""
        text_lower = text.lower()
        emotions_detected = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            
            for word in patterns['words']:
                count = len(re.findall(r'\b' + word + r'\b', text_lower))
                if count > 0:
                    score += count * 0.3
            
            for phrase in patterns['phrases']:
                if phrase in text_lower:
                    score += 0.5
            
            intensifiers = ['very', 'extremely', 'really', 'so', 'truly', 'deeply']
            for intensifier in intensifiers:
                for word in patterns['words']:
                    if f"{intensifier} {word}" in text_lower:
                        score += 0.2
            
            if score > 0:
                emotions_detected[emotion] = round(score, 2)
        
        if not emotions_detected:
            vader_scores = self.analyze_sentiment_vader(text)
            if vader_scores['compound'] >= 0.5:
                emotions_detected['joy'] = 0.5
            elif vader_scores['compound'] <= -0.5:
                emotions_detected['sadness'] = 0.5
            elif vader_scores['compound'] > 0.1:
                emotions_detected['trust'] = 0.3
            elif vader_scores['compound'] < -0.1:
                emotions_detected['fear'] = 0.3
            else:
                emotions_detected['neutral'] = 0.5
        
        return emotions_detected
    
    def extract_key_phrases(self, text, num_phrases=5):
        """Extract key phrases using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(max_features=num_phrases, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            return list(feature_names)
        except:
            words = text.split()
            word_counts = Counter(words)
            return [word for word, _ in word_counts.most_common(num_phrases) if len(word) > 3]
    
    def assess_crisis_risk(self, text):
        """Assess if text indicates crisis risk"""
        crisis_indicators = {
            'high': ['suicide', 'kill myself', 'end my life', 'want to die', 
                    'self-harm', 'hurt myself', 'no reason to live'],
            'medium': ['hopeless', 'can\'t go on', 'giving up', 'no way out',
                      'tired of living', 'better off dead'],
            'low': ['sad', 'depressed', 'lonely', 'alone', 'struggling']
        }
        
        text_lower = text.lower()
        risk_level = 'none'
        indicators_found = []
        
        for level, indicators in crisis_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    indicators_found.append(indicator)
                    if level == 'high':
                        risk_level = 'HIGH'
                    elif level == 'medium' and risk_level != 'HIGH':
                        risk_level = 'MEDIUM'
                    elif level == 'low' and risk_level not in ['HIGH', 'MEDIUM']:
                        risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'indicators': indicators_found
        }
    
    def get_wellness_recommendations(self, dominant_emotion, text):
        """Get personalized wellness recommendations"""
        emotion_data = TEXT_EMOTION_MAPPING.get(dominant_emotion, {
            'status': '😐 Neutral',
            'suggestion': 'Your text appears neutral. This is a good baseline.',
            'activities': ['Practice mindfulness', 'Read something inspiring', 'Take a walk'],
            'color': '#808080',
            'wellness_tip': 'A balanced state is perfect for self-reflection.',
            'icon': '😐',
            'category': 'Neutral'
        })
        
        return emotion_data

# Initialize text analyzer
text_analyzer = TextEmotionAnalyzer()

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
    .crisis-high {
        background-color: #EF444420;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .crisis-medium {
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


##create_weekly_pattern function##
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

##create_emotion_radar function##
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
        if 'vader_scores' in entry and isinstance(entry['vader_scores'], dict):
            sentiment_sum += entry['vader_scores'].get('compound', 0)
        
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
        sentiment = f"{entry.get('vader_scores', {}).get('compound', 0):.2f}" if isinstance(entry.get('vader_scores'), dict) else 'N/A'
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

# ==================== TEXT ANALYSIS FUNCTIONS ====================
def show_text_analysis():
    st.markdown("## 📝 Text Emotion Analysis")
    st.markdown("Analyze your text to understand emotional content and get mental health insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter your text for analysis:",
            height=200,
            placeholder="Type or paste your thoughts, journal entries, or any text here...",
            help="Share your thoughts, feelings, or any text you'd like to analyze"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            analyze_btn = st.button("🔍 Analyze Text", use_container_width=True, type="primary")
        with col_b:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
            
        if clear_btn:
            text_input = ""
            st.rerun()
    
    with col2:
        st.markdown("### ⚙️ Analysis Options")
        analyze_depth = st.radio(
            "Analysis Depth",
            ["Basic", "Advanced", "Clinical"],
            help="Choose analysis depth level"
        )
        
        st.markdown("### 📋 Context (Optional)")
        context = st.selectbox(
            "Text Context",
            ["General", "Journal Entry", "Therapy Notes", "Social Media", "Email", "Other"],
            help="Select the context of your text"
        )
        
        save_to_history = st.checkbox("Save to History", value=True, help="Save this analysis to your history")
    
    if analyze_btn and text_input:
        with st.spinner("Performing advanced text analysis..."):
            vader_scores = text_analyzer.analyze_sentiment_vader(text_input)
            textblob_scores = text_analyzer.analyze_sentiment_textblob(text_input)
            emotions_detected = text_analyzer.detect_emotions(text_input)
            key_phrases = text_analyzer.extract_key_phrases(text_input)
            crisis_risk = text_analyzer.assess_crisis_risk(text_input)
            
            if emotions_detected:
                dominant_emotion = max(emotions_detected, key=emotions_detected.get)
                dominant_score = emotions_detected[dominant_emotion]
            else:
                if vader_scores['compound'] >= 0.5:
                    dominant_emotion = 'joy'
                elif vader_scores['compound'] <= -0.5:
                    dominant_emotion = 'sadness'
                elif vader_scores['compound'] > 0.1:
                    dominant_emotion = 'trust'
                elif vader_scores['compound'] < -0.1:
                    dominant_emotion = 'fear'
                else:
                    dominant_emotion = 'neutral'
                dominant_score = 0.5
            
            wellness = text_analyzer.get_wellness_recommendations(dominant_emotion, text_input)
            
            word_count = len(text_input.split())
            char_count = len(text_input)
            sentence_count = len([s for s in text_input.split('.') if s.strip()])
            avg_word_length = char_count / word_count if word_count > 0 else 0
            
            text_analysis_result = {
                'type': 'text_analysis',
                'dominant_emotion': dominant_emotion,
                'dominant_score': dominant_score,
                'emotions_detected': emotions_detected,
                'vader_scores': vader_scores,
                'textblob_scores': textblob_scores,
                'key_phrases': key_phrases,
                'crisis_risk': crisis_risk,
                'wellness': wellness,
                'statistics': {
                    'word_count': word_count,
                    'char_count': char_count,
                    'sentence_count': sentence_count,
                    'avg_word_length': avg_word_length
                },
                'context': context,
                'text_preview': text_input[:200] + "..." if len(text_input) > 200 else text_input,
                'full_text': text_input,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            display_text_analysis_results(text_analysis_result, analyze_depth)
            
            if save_to_history:
                st.session_state.analysis_history.append(text_analysis_result)
                if save_analysis_to_history(st.session_state.username, text_analysis_result):
                    st.success("✅ Analysis saved to your history!")
    
    with st.expander("📋 Try these examples"):
        examples = [
            "I'm feeling really happy and grateful for all the good things in my life today!",
            "I'm so frustrated and angry about what happened at work. Nothing ever goes right.",
            "Feeling anxious and worried about the future. So many uncertainties.",
            "I don't know how to feel. Everything is just okay, I guess.",
            "I can't believe it! This is absolutely amazing and unexpected!"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Try Example {i+1}", key=f"example_{i}"):
                text_input = example
                st.rerun()

def display_text_analysis_results(result, depth):
    """Display text analysis results with visualizations"""
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    crisis_risk = result['crisis_risk']
    if crisis_risk['risk_level'] != 'none':
        risk_color = {
            'HIGH': '#EF4444',
            'MEDIUM': '#F59E0B',
            'LOW': '#FBBF24'
        }.get(crisis_risk['risk_level'], '#808080')
        
        risk_class = f"crisis-{crisis_risk['risk_level'].lower()}"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h4 style="color: {risk_color}; margin: 0;">⚠️ {crisis_risk['risk_level']} RISK DETECTED</h4>
            <p style="margin: 0.5rem 0 0 0;">
                Your text contains indicators that may require attention. Please reach out to a mental health professional 
                or call a crisis helpline if you're in distress.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📞 Crisis Resources"):
            st.markdown("""
            **National Suicide Prevention Lifeline:** 988 or 1-800-273-8255  
            **Crisis Text Line:** Text HOME to 741741  
            **SAMHSA National Helpline:** 1-800-662-4357  
            **Emergency:** 911
            """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    wellness = result['wellness']
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: {wellness['color']}20; 
                    border-radius: 10px; border-left: 5px solid {wellness['color']};">
            <h3 style="color: {wellness['color']}; margin: 0;">{wellness['icon']}</h3>
            <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{result['dominant_emotion'].title()}</p>
            <p style="margin: 0; font-size: 0.8rem;">Primary Emotion</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Sentiment Score", f"{result['vader_scores']['compound']:.2f}")
    
    with col3:
        st.metric("Word Count", result['statistics']['word_count'])
    
    with col4:
        st.metric("Subjectivity", f"{result['textblob_scores']['subjectivity']:.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎭 Emotion Distribution")
        
        emotions_df = pd.DataFrame([
            {'Emotion': emotion.capitalize(), 'Intensity': score}
            for emotion, score in result['emotions_detected'].items()
        ]).sort_values('Intensity', ascending=True)
        
        if not emotions_df.empty:
            fig = px.bar(
                emotions_df,
                x='Intensity',
                y='Emotion',
                orientation='h',
                color='Intensity',
                color_continuous_scale='Viridis',
                title="Detected Emotions Intensity"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Sentiment Analysis")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='VADER',
            x=['Positive', 'Neutral', 'Negative', 'Compound'],
            y=[result['vader_scores']['pos'], 
               result['vader_scores']['neu'],
               result['vader_scores']['neg'],
               result['vader_scores']['compound']],
            marker_color=['#22C55E', '#6B7280', '#EF4444', '#6366F1']
        ))
        fig.update_layout(title="VADER Sentiment Scores", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    if depth in ["Advanced", "Clinical"]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔑 Key Phrases")
            if result['key_phrases']:
                for phrase in result['key_phrases']:
                    st.markdown(f"- {phrase}")
            else:
                st.info("No key phrases detected")
        
        with col2:
            st.markdown("### 📈 Text Statistics")
            stats = result['statistics']
            st.markdown(f"""
            - **Word Count:** {stats['word_count']}
            - **Character Count:** {stats['char_count']}
            - **Sentence Count:** {stats['sentence_count']}
            - **Avg Word Length:** {stats['avg_word_length']:.1f} chars
            - **TextBlob Polarity:** {result['textblob_scores']['polarity']:.2f}
            - **TextBlob Subjectivity:** {result['textblob_scores']['subjectivity']:.2f}
            """)
    
    if depth == "Clinical":
        st.markdown("### 🏥 Clinical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Emotional Indicators:**")
            for emotion, score in result['emotions_detected'].items():
                color = TEXT_EMOTION_MAPPING.get(emotion, {}).get('color', '#808080')
                st.markdown(f"- <span style='color:{color}'>●</span> {emotion.title()}: {score:.2f}", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Risk Assessment:**")
            risk_level = result['crisis_risk']['risk_level']
            risk_color = {
                'HIGH': '#EF4444',
                'MEDIUM': '#F59E0B',
                'LOW': '#FBBF24',
                'none': '#22C55E'
            }.get(risk_level, '#808080')
            
            st.markdown(f"- **Crisis Risk:** <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
            if result['crisis_risk']['indicators']:
                st.markdown(f"- **Indicators:** {', '.join(result['crisis_risk']['indicators'])}")
    
    st.markdown("### 🌱 Wellness Insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="wellness-tip" style="border-left-color: {wellness['color']};">
            <h4 style="color: {wellness['color']};">{wellness['status']}</h4>
            <p><strong>Suggestion:</strong> {wellness['suggestion']}</p>
            <p><strong>Wellness Tip:</strong> {wellness['wellness_tip']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Recommended Activities:**")
        for activity in wellness['activities']:
            if st.button(f"✅ {activity}", key=f"text_act_{activity}"):
                st.success(f"Great! Try to {activity.lower()} now.")
    
    with st.expander("📝 View Full Text"):
        st.markdown(result['full_text'])

# ==================== FACIAL ANALYSIS FUNCTIONS ====================
def show_emotion_analysis():
    st.markdown("## 🎭 Facial Emotion Analysis")
    st.markdown("Analyze your facial expression to get mental health insights")
    
    if model is None:
        st.error("⚠️ Facial emotion recognition model not found. Please ensure 'emotiondetector.h5' exists in the project directory.")
        st.info("Text analysis is still available in the 'Text Analysis' tab.")
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
                <p>Start by analyzing your emotions using the Facial Analysis or Text Analysis page!</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    analysis_type = st.selectbox(
        "Filter by Analysis Type",
        ["All", "Facial Analysis", "Text Analysis"]
    )
    
    filtered_history = []
    for entry in st.session_state.analysis_history:
        if analysis_type == "All":
            filtered_history.append(entry)
        elif analysis_type == "Facial Analysis" and entry.get('type') != 'text_analysis':
            filtered_history.append(entry)
        elif analysis_type == "Text Analysis" and entry.get('type') == 'text_analysis':
            filtered_history.append(entry)
    
    for i, entry in enumerate(reversed(filtered_history)):
        if entry.get('type') == 'text_analysis':
            wellness = entry.get('wellness', {})
            color = wellness.get('color', '#808080')
            
            with st.expander(f"📝 Text Analysis {len(filtered_history)-i}: {entry['timestamp']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Text Preview:** {entry.get('text_preview', 'N/A')}")
                    st.markdown(f"**Primary Emotion:** <span style='color:{color}'>{entry['dominant_emotion'].upper()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Sentiment Score:** {entry['vader_scores']['compound']:.2f}")
                    st.markdown(f"**Word Count:** {entry['statistics']['word_count']}")
                    st.markdown(f"**Context:** {entry.get('context', 'General')}")
                
                with col2:
                    if st.button(f"View Details", key=f"view_text_{i}"):
                        st.session_state.current_text_analysis = entry
                        st.rerun()
                    
                    if st.button(f"Delete", key=f"del_text_{i}"):
                        st.session_state.analysis_history.remove(entry)
                        st.rerun()
        else:
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
            ["All", "Facial Only", "Text Only", "Combined"]
        )
    with col2:
        date_range = st.selectbox(
            "Date Range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
        )
    
    # Filter data based on analysis type
    filtered_data = []
    for entry in st.session_state.analysis_history:
        if analysis_type_filter == "All":
            filtered_data.append(entry)
        elif analysis_type_filter == "Facial Only" and entry.get('type') != 'text_analysis':
            filtered_data.append(entry)
        elif analysis_type_filter == "Text Only" and entry.get('type') == 'text_analysis':
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
                # Check if timestamp exists
                if 'timestamp' in e:
                    try:
                        # Extract date part (first 10 characters: YYYY-MM-DD)
                        date_str = e['timestamp'][:10]
                        entry_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if entry_date >= cutoff:
                            date_filtered.append(e)
                    except (ValueError, KeyError, TypeError):
                        # If date parsing fails, include the entry anyway
                        date_filtered.append(e)
                else:
                    # If no timestamp, include the entry with a warning
                    date_filtered.append(e)
            
            filtered_data = date_filtered
        except Exception as e:
            st.warning(f"Date filtering issue: {str(e)}")
    else:
        # For "All time", find the earliest date
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
    
    # Separate facial and text data with safe access
    facial_data = []
    text_data = []
    
    for entry in filtered_data:
        if entry.get('type') == 'text_analysis':
            text_data.append(entry)
        else:
            facial_data.append(entry)
    
    # Create DataFrames safely
    facial_df = pd.DataFrame(facial_data) if facial_data else pd.DataFrame()
    text_df = pd.DataFrame(text_data) if text_data else pd.DataFrame()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyses", len(filtered_data))
    with col2:
        st.metric("Facial Analyses", len(facial_data))
    with col3:
        st.metric("Text Analyses", len(text_data))
    with col4:
        try:
            unique_days = len(set([e.get('timestamp', '')[:10] for e in filtered_data if e.get('timestamp')]))
        except:
            unique_days = len(filtered_data)
        st.metric("Active Days", unique_days)
    
    # Facial vs Text Comparison
    if facial_data and text_data:
        st.markdown("### 📊 Facial vs Text Analysis Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not facial_df.empty and 'dominant_emotion' in facial_df.columns:
                facial_counts = facial_df['dominant_emotion'].value_counts()
                fig_facial = px.pie(
                    values=facial_counts.values,
                    names=[e.capitalize() for e in facial_counts.index],
                    title="Facial Emotion Distribution",
                    color_discrete_sequence=[MENTAL_HEALTH_MAPPING.get(e, {}).get('color', '#808080') 
                                            for e in facial_counts.index]
                )
                st.plotly_chart(fig_facial, use_container_width=True)
        
        with col2:
            if not text_df.empty and 'dominant_emotion' in text_df.columns:
                text_counts = text_df['dominant_emotion'].value_counts()
                fig_text = px.pie(
                    values=text_counts.values,
                    names=[e.capitalize() for e in text_counts.index],
                    title="Text Emotion Distribution",
                    color_discrete_sequence=[TEXT_EMOTION_MAPPING.get(e, {}).get('color', '#808080') 
                                            for e in text_counts.index]
                )
                st.plotly_chart(fig_text, use_container_width=True)
    
    # Facial Analysis Charts
    if facial_data:
        st.markdown("### 🎭 Facial Analysis Timeline")
        
        # Add timestamp to facial data for timeline
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
        
        # Calculate text metrics safely
        avg_sentiment = 0
        avg_word_count = 0
        crisis_count = 0
        
        for entry in text_data:
            # Sentiment score
            if 'vader_scores' in entry and isinstance(entry['vader_scores'], dict):
                avg_sentiment += entry['vader_scores'].get('compound', 0)
            
            # Word count
            if 'statistics' in entry and isinstance(entry['statistics'], dict):
                avg_word_count += entry['statistics'].get('word_count', 0)
            
            # Crisis count
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
        
        # Text emotion distribution
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
    
    # ==================== PDF REPORT GENERATION SECTION ====================
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
            ["Comprehensive Report", "Facial Analysis Only", "Text Analysis Only", "Summary Report"],
            key="report_type"
        )
    
    with col3:
        include_charts = st.checkbox("Include Charts in Report", value=True, key="include_charts")
    
    # Prepare data for report based on report type
    report_data = []
    if report_type == "Facial Analysis Only":
        report_data = facial_data
        report_df = facial_df.copy() if not facial_df.empty else pd.DataFrame()
    elif report_type == "Text Analysis Only":
        report_data = text_data
        report_df = pd.DataFrame()  # Text data uses different PDF generator
    else:
        report_data = filtered_data
        # Create combined dataframe for facial data only (for the PDF report function)
        report_df = facial_df.copy() if not facial_df.empty else pd.DataFrame()
    
    # Generate report button
    if st.button("📄 Generate Hospital-Grade PDF Report", use_container_width=True, type="primary"):
        if not report_data:
            st.error("No data available for the selected report type!")
        else:
            with st.spinner("Generating professional clinical report..."):
                try:
                    # Format dates for report
                    start_date_str = start_date.strftime("%Y-%m-%d") if start_date else "N/A"
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    
                    # Create title based on report type
                    final_title = f"{report_title} - {report_type}"
                    
                    # Add analysis type indicator to title
                    if report_type == "Facial Analysis Only":
                        final_title += " (Facial Data)"
                    elif report_type == "Text Analysis Only":
                        final_title += " (Text Data)"
                    
                    # Generate PDF based on report type
                    pdf_bytes = None
                    
                    if report_type == "Text Analysis Only" and text_data:
                        # Use text-specific PDF generator
                        pdf_bytes = generate_text_pdf_report(
                            text_data,
                            st.session_state.username,
                            start_date_str,
                            end_date_str,
                            final_title
                        )
                    elif not report_df.empty:
                        # Use facial PDF generator for all other report types
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
                        # Create filename
                        filename = f"NeuroWell_Report_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        
                        # Download button
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
                        # Offer alternative options
                        if text_data and report_type != "Text Analysis Only":
                            st.info("📝 Text analysis data detected. Try selecting 'Text Analysis Only' report type.")
                        if facial_data and report_type != "Facial Analysis Only":
                            st.info("🎭 Facial analysis data detected. Try selecting 'Facial Analysis Only' report type.")
                
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.exception(e)  # This will show the full error for debugging
    
    # Additional export options
    st.markdown("### 📊 Additional Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Data as CSV", use_container_width=True):
            # Prepare data for CSV export
            export_data = []
            for entry in filtered_data:
                if entry.get('type') == 'text_analysis':
                    export_data.append({
                        'Date': entry.get('timestamp', 'N/A')[:10],
                        'Time': entry.get('timestamp', 'N/A')[11:16] if len(entry.get('timestamp', '')) > 16 else 'N/A',
                        'Type': 'Text Analysis',
                        'Primary Emotion': entry.get('dominant_emotion', 'N/A'),
                        'Sentiment Score': entry.get('vader_scores', {}).get('compound', 0) if isinstance(entry.get('vader_scores'), dict) else 0,
                        'Word Count': entry.get('statistics', {}).get('word_count', 0) if isinstance(entry.get('statistics'), dict) else 0,
                        'Crisis Risk': entry.get('crisis_risk', {}).get('risk_level', 'none') if isinstance(entry.get('crisis_risk'), dict) else 'none'
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
        Your comprehensive mental health companion that uses both facial emotion recognition 
        and text analysis to help you understand and manage your emotional well-being.
        
        ### 🌟 Key Features:
        - **Facial Emotion Analysis**: Understand your emotional state from facial expressions
        - **Text Emotion Analysis**: Analyze journal entries, thoughts, and messages
        - **Mental Health Insights**: Get personalized suggestions based on your emotions
        - **Track Your Journey**: Monitor your emotional patterns over time
        - **Advanced Analytics**: Visualize your emotional trends with multiple chart types
        - **Wellness Tips**: Access curated mental health resources
        
        ### 📊 Quick Start:
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎭 Try Facial Analysis", use_container_width=True):
                st.session_state.current_page = "Facial Analysis"
                st.rerun()
        
        with col_b:
            if st.button("📝 Try Text Analysis", use_container_width=True):
                st.session_state.current_page = "Text Analysis"
                st.rerun()
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; animation: float 3s ease-in-out infinite;">
            {get_logo_html('xlarge')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌱 Today's Tip")
        st.info("Your emotions are valid messengers, whether expressed through your face or your words. Listen to what they're telling you.")
        
        if st.session_state.analysis_history:
            st.markdown("### 📊 Your Stats")
            total = len(st.session_state.analysis_history)
            facial_count = len([e for e in st.session_state.analysis_history if e.get('type') != 'text_analysis'])
            text_count = len([e for e in st.session_state.analysis_history if e.get('type') == 'text_analysis'])
            
            st.metric("Total Analyses", total)
            st.metric("Facial Analyses", facial_count)
            st.metric("Text Analyses", text_count)

# ==================== MENTAL HEALTH TIPS ====================
def show_mental_health_tips():
    st.markdown("## 💚 Mental Health Tips")
    st.markdown("Personalized wellness tips based on your emotional state")
    
    tab1, tab2 = st.tabs(["🎭 Facial Emotions", "📝 Text Emotions"])
    
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
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Facial Analysis", "Text Analysis", "Mental Health Tips", "History", "Analytics", "Profile"],
            icons=["house", "camera", "pencil-square", "heart", "clock-history", "graph-up", "person"],
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
        show_text_analysis()
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