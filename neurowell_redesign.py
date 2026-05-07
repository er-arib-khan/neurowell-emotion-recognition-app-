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

try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# ==================== COLOR PALETTES ====================
LIGHT_COLORS = {
    'primary': '#6366F1',
    'primary_dark': '#4F46E5',
    'primary_light': '#818CF8',
    'secondary': '#10B981',
    'accent': '#F59E0B',
    'danger': '#EF4444',
    'warning': '#F97316',
    'success': '#22C55E',
    'info': '#3B82F6',
    'bg': '#F9FAFB',
    'surface': '#FFFFFF',
    'surface2': '#F3F4F6',
    'text': '#1F2937',
    'text_secondary': '#6B7280',
    'border': '#E5E7EB',
    'shadow': 'rgba(0,0,0,0.08)',
    'glass': 'rgba(255,255,255,0.7)',
    'glass_border': 'rgba(255,255,255,0.5)',
}

DARK_COLORS = {
    'primary': '#818CF8',
    'primary_dark': '#6366F1',
    'primary_light': '#A5B4FC',
    'secondary': '#34D399',
    'accent': '#FBBF24',
    'danger': '#F87171',
    'warning': '#FB923C',
    'success': '#4ADE80',
    'info': '#60A5FA',
    'bg': '#111827',
    'surface': '#1F2937',
    'surface2': '#374151',
    'text': '#F9FAFB',
    'text_secondary': '#9CA3AF',
    'border': '#374151',
    'shadow': 'rgba(0,0,0,0.4)',
    'glass': 'rgba(31,41,55,0.8)',
    'glass_border': 'rgba(255,255,255,0.1)',
}

EMOTION_COLORS = {
    'angry': '#EF4444', 'disgust': '#10B981', 'fear': '#8B5CF6',
    'happy': '#F59E0B', 'neutral': '#6B7280', 'sad': '#3B82F6', 'surprise': '#EC4899'
}

VOICE_EMOTION_MAPPING = {
    'calm': {'status': '😌 Calm / Relaxed', 'suggestion': 'Your voice indicates a calm state. Excellent for mindfulness and clear thinking.', 'activities': ['Practice meditation', 'Engage in deep breathing', 'Listen to soothing music'], 'color': '#4A90E2', 'wellness_tip': 'Calmness is a superpower. Use it to make thoughtful decisions.', 'icon': '😌', 'category': 'Positive', 'clinical_term': 'Relaxed State', 'intervention': 'Mindfulness-based stress reduction'},
    'happy': {'status': '😊 Happy / Energetic', 'suggestion': 'Your voice conveys happiness! Channel this positive energy into creative activities.', 'activities': ['Share your joy with others', 'Start a creative project', 'Exercise'], 'color': '#4CAF50', 'wellness_tip': 'Happiness is contagious. Spread it to those around you.', 'icon': '😊', 'category': 'Positive', 'clinical_term': 'Elevated Mood', 'intervention': 'Positive psychology interventions'},
    'sad': {'status': '😔 Sad / Low Energy', 'suggestion': 'Your voice suggests low mood. Be gentle with yourself today.', 'activities': ['Reach out to a friend', 'Practice self-care', 'Gentle stretching'], 'color': '#4169E1', 'wellness_tip': "It's okay to not be okay. Give yourself permission to feel.", 'icon': '😔', 'category': 'Negative', 'clinical_term': 'Depressed Mood', 'intervention': 'Behavioral activation therapy'},
    'angry': {'status': '😠 Angry / Agitated', 'suggestion': 'Your voice shows signs of agitation. Take a moment to breathe.', 'activities': ['Count to ten', 'Go for a walk', 'Write down your feelings'], 'color': '#EF4444', 'wellness_tip': 'Anger is a signal, not a solution. What is it telling you?', 'icon': '😠', 'category': 'Negative', 'clinical_term': 'Irritability', 'intervention': 'Anger management techniques'},
    'fearful': {'status': '😨 Fearful / Anxious', 'suggestion': 'Your voice indicates anxiety. Practice grounding techniques.', 'activities': ['Box breathing', 'Grounding exercises', 'Talk to someone'], 'color': '#8B5CF6', 'wellness_tip': 'Fear is future-focused. Bring yourself back to the present.', 'icon': '😨', 'category': 'Negative', 'clinical_term': 'Anxiety Response', 'intervention': 'Anxiety management and grounding'},
    'neutral': {'status': '😐 Neutral / Balanced', 'suggestion': 'Your voice sounds balanced. Perfect for focused work.', 'activities': ['Start a task', 'Read something new', 'Practice mindfulness'], 'color': '#6B7280', 'wellness_tip': 'Balance is the key to sustainable well-being.', 'icon': '😐', 'category': 'Neutral', 'clinical_term': 'Baseline State', 'intervention': 'Maintain current coping strategies'},
    'surprised': {'status': '😲 Surprised / Alert', 'suggestion': "Your voice shows surprise or alertness. Process what you're experiencing.", 'activities': ['Pause and reflect', 'Journal about it', 'Share your experience'], 'color': '#F59E0B', 'wellness_tip': 'Surprise opens the door to curiosity and learning.', 'icon': '😲', 'category': 'Neutral', 'clinical_term': 'Startle Response', 'intervention': 'Mindful observation'}
}

TEXT_EMOTION_MAPPING = {
    'joy': {'status': '😊 Joyful / Positive', 'suggestion': 'Your text conveys positivity! Great for mental well-being.', 'activities': ['Express gratitude', 'Share your positivity', 'Engage in creative writing'], 'color': '#4CAF50', 'wellness_tip': 'Positive self-talk reinforces emotional resilience.', 'icon': '😊', 'category': 'Positive'},
    'sadness': {'status': '😔 Sadness / Melancholy', 'suggestion': 'Your text shows signs of sadness. Consider reaching out to someone you trust.', 'activities': ['Self-care routine', 'Connect with a loved one', 'Journal your feelings'], 'color': '#4169E1', 'wellness_tip': "It's okay to feel sad. Acknowledging it is the first step.", 'icon': '😔', 'category': 'Negative'},
    'anger': {'status': '😤 Anger / Frustration', 'suggestion': 'Notice the anger in your text. Take a moment to breathe before responding.', 'activities': ['Deep breathing', 'Physical exercise', 'Count to ten'], 'color': '#EF4444', 'wellness_tip': 'Anger often signals unmet needs. What might you need right now?', 'icon': '😠', 'category': 'Negative'},
    'fear': {'status': '😨 Fear / Anxiety', 'suggestion': 'Your text suggests anxiety. Practice grounding techniques.', 'activities': ['Box breathing', 'Progressive muscle relaxation', 'Grounding exercises'], 'color': '#800080', 'wellness_tip': 'Fear is future-focused. Bring yourself back to the present moment.', 'icon': '😰', 'category': 'Negative'},
    'surprise': {'status': '😲 Surprise / Shock', 'suggestion': 'Your text indicates surprise. Take time to process new information.', 'activities': ['Pause and reflect', 'Journal about it', 'Discuss with someone'], 'color': '#FFA500', 'wellness_tip': 'Surprise can be an invitation to curiosity.', 'icon': '😲', 'category': 'Neutral'},
    'trust': {'status': '🤝 Trust / Confidence', 'suggestion': 'Your text shows trust and confidence. This builds strong relationships.', 'activities': ['Build on this trust', 'Help others', 'Share your confidence'], 'color': '#3B82F6', 'wellness_tip': 'Trust is the foundation of emotional well-being.', 'icon': '🤝', 'category': 'Positive'},
    'anticipation': {'status': '🔮 Anticipation / Hope', 'suggestion': "You're looking forward to something. Channel this positive energy.", 'activities': ['Plan ahead', 'Set goals', 'Visualize success'], 'color': '#F59E0B', 'wellness_tip': 'Anticipation can be a source of motivation and hope.', 'icon': '🔮', 'category': 'Positive'}
}

MENTAL_HEALTH_MAPPING = {
    'angry': {'status': '😤 High Stress / Irritability', 'suggestion': 'Consider deep breathing or stepping away. Try the 5-4-3-2-1 grounding technique.', 'activities': ['Take 5 deep breaths', 'Go for a short walk', 'Listen to calming music'], 'color': '#FF4B4B', 'wellness_tip': 'Anger often signals unmet needs or boundaries. What might you need right now?', 'icon': '😠', 'category': 'Negative', 'clinical_term': 'Elevated Irritability', 'intervention': 'Cognitive Behavioral Therapy techniques for anger management'},
    'disgust': {'status': '🤢 Discomfort / Aversion', 'suggestion': 'This might indicate something in your environment is bothering you. Identify the source.', 'activities': ['Fresh air break', 'Clean/organize your space', 'Mindful observation'], 'color': '#9ACD32', 'wellness_tip': 'Disgust can be a protective emotion. Honor what your body is telling you.', 'icon': '🤢', 'category': 'Negative', 'clinical_term': 'Aversive Response', 'intervention': 'Exposure therapy and sensory integration'},
    'fear': {'status': '😰 Anxiety / Apprehension', 'suggestion': 'Practice grounding techniques. Name 5 things you see, 4 you can touch, 3 you hear.', 'activities': ['Box breathing (4-4-4-4)', 'Progressive muscle relaxation', 'Talk to someone you trust'], 'color': '#800080', 'wellness_tip': 'Fear is future-focused. Bring yourself back to the present moment.', 'icon': '😨', 'category': 'Negative', 'clinical_term': 'Anxiety Response', 'intervention': 'Anxiety management techniques and grounding exercises'},
    'happy': {'status': '😊 Positive / Content', 'suggestion': 'Great! This is a good state for productivity and connection. Share your positivity.', 'activities': ['Express gratitude', 'Help someone else', 'Engage in a creative activity'], 'color': '#4CAF50', 'wellness_tip': 'Savor this moment. What contributed to your happiness today?', 'icon': '😊', 'category': 'Positive', 'clinical_term': 'Euthymic State', 'intervention': 'Positive psychology interventions and gratitude practices'},
    'neutral': {'status': '😐 Calm / Balanced', 'suggestion': "You're in a balanced state. Perfect for mindfulness or starting a new task.", 'activities': ["Start that project you've been putting off", 'Practice mindfulness', 'Read or learn something new'], 'color': '#808080', 'wellness_tip': "Neutral doesn't mean empty. It's a peaceful baseline for well-being.", 'icon': '😐', 'category': 'Neutral', 'clinical_term': 'Baseline Affective State', 'intervention': 'Maintenance of current coping strategies'},
    'sad': {'status': '😔 Low Mood / Melancholy', 'suggestion': 'Be gentle with yourself. This is a valid emotion that deserves acknowledgment.', 'activities': ['Self-care routine', 'Connect with a loved one', 'Gentle exercise or stretching'], 'color': '#4169E1', 'wellness_tip': 'Sadness slows us down to process and heal. What do you need right now?', 'icon': '😔', 'category': 'Negative', 'clinical_term': 'Depressed Mood', 'intervention': 'Behavioral activation and social connection'},
    'surprise': {'status': '😲 Alert / Stimulated', 'suggestion': 'Your nervous system is activated. Take a moment to assess.', 'activities': ['Pause and breathe', 'Journal about what surprised you', 'Channel the energy creatively'], 'color': '#FFA500', 'wellness_tip': 'Surprise can be an invitation to curiosity. What can you learn?', 'icon': '😲', 'category': 'Neutral', 'clinical_term': 'Startle Response', 'intervention': 'Mindful observation and cognitive reframing'}
}

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

DAILY_QUOTES = [
    ("You don't have to be positive all the time. It's perfectly okay to feel sad, angry, annoyed, frustrated, scared or anxious.", "Lori Deschene"),
    ("Mental health is not a destination but a process.", "Noam Shpancer"),
    ("Self-care is not self-indulgence, it is self-preservation.", "Audre Lorde"),
    ("You are allowed to be both a masterpiece and a work in progress simultaneously.", "Sophia Bush"),
    ("There is hope, even when your brain tells you there isn't.", "John Green"),
    ("Not until we are lost do we begin to understand ourselves.", "Henry David Thoreau"),
    ("The greatest glory in living lies not in never falling, but in rising every time we fall.", "Nelson Mandela"),
]

ACHIEVEMENTS = {
    'first_analysis': {'name': 'First Step', 'icon': '🌱', 'desc': 'Complete your first analysis', 'threshold': 1},
    'ten_analyses': {'name': 'Explorer', 'icon': '🔍', 'desc': 'Complete 10 analyses', 'threshold': 10},
    'voice_master': {'name': 'Voice Master', 'icon': '🎤', 'desc': 'Complete 5 voice analyses', 'threshold': 5},
    'journal_pro': {'name': 'Journaling Pro', 'icon': '📝', 'desc': 'Complete 5 text analyses', 'threshold': 5},
    'streak_3': {'name': '3-Day Streak', 'icon': '🔥', 'desc': 'Analyze emotions 3 days in a row', 'threshold': 3},
    'streak_7': {'name': 'Week Warrior', 'icon': '⚡', 'desc': 'Analyze emotions 7 days in a row', 'threshold': 7},
}

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="NeuroWell — Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
def init_session():
    defaults = {
        'logged_in': False, 'username': None, 'current_page': 'Home',
        'analysis_history': [], 'profile_image': None,
        'current_text_analysis': None, 'current_voice_analysis': None,
        'recording': False, 'dark_mode': False, 'onboarding_done': False,
        'font_size': 'medium', 'streak': 0, 'last_analysis_date': None,
        'achievements': [], 'notification_count': 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ==================== THEME SYSTEM ====================
def get_colors():
    return DARK_COLORS if st.session_state.dark_mode else LIGHT_COLORS

def get_font_size():
    sizes = {'small': '13px', 'medium': '15px', 'large': '17px'}
    return sizes.get(st.session_state.font_size, '15px')

def get_theme_css():
    C = get_colors()
    fs = get_font_size()
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --primary: {C['primary']};
        --primary-dark: {C['primary_dark']};
        --primary-light: {C['primary_light']};
        --secondary: {C['secondary']};
        --accent: {C['accent']};
        --danger: {C['danger']};
        --warning: {C['warning']};
        --success: {C['success']};
        --info: {C['info']};
        --bg: {C['bg']};
        --surface: {C['surface']};
        --surface2: {C['surface2']};
        --text: {C['text']};
        --text-secondary: {C['text_secondary']};
        --border: {C['border']};
        --shadow: {C['shadow']};
        --glass: {C['glass']};
        --glass-border: {C['glass_border']};
        --font-size: {fs};
    }}

    * {{ font-family: 'Inter', sans-serif !important; font-size: var(--font-size); }}

    html, body, .stApp {{
        background: var(--bg) !important;
        color: var(--text) !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }}
    section[data-testid="stSidebar"] > div {{
        background: var(--surface) !important;
    }}

    /* Main content area */
    .main .block-container {{
        padding: 1.5rem 2rem 3rem 2rem !important;
        max-width: 1400px !important;
    }}

    /* Cards & Glass morphism */
    .nw-card {{
        background: var(--glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px var(--shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .nw-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 16px 48px var(--shadow);
    }}

    .nw-card-flat {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px var(--shadow);
    }}

    /* Emotion Display Card */
    .emotion-hero {{
        text-align: center;
        padding: 2.5rem 1.5rem;
        border-radius: 24px;
        position: relative;
        overflow: hidden;
    }}
    .emotion-hero::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }}
    @keyframes shimmer {{
        0%,100% {{ transform: rotate(0deg); }}
        50% {{ transform: rotate(180deg); }}
    }}

    /* Metric Cards */
    .metric-pill {{
        background: linear-gradient(135deg, var(--primary)15, var(--primary)08);
        border: 1px solid var(--primary)30;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }}
    .metric-pill .value {{
        font-size: 2rem;
        font-weight: 800;
        color: var(--primary);
        display: block;
        line-height: 1;
    }}
    .metric-pill .label {{
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }}

    /* Quick Action Cards */
    .action-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.75rem 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }}
    .action-card:hover {{
        border-color: var(--primary);
        transform: translateY(-4px);
        box-shadow: 0 20px 60px var(--shadow);
    }}
    .action-card .icon {{
        font-size: 2.5rem;
        display: block;
        margin-bottom: 0.75rem;
    }}
    .action-card .title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.5rem;
    }}
    .action-card .desc {{
        font-size: 0.85rem;
        color: var(--text-secondary);
    }}

    /* Badge */
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .badge-positive {{ background: #22C55E20; color: #22C55E; }}
    .badge-negative {{ background: #EF444420; color: #EF4444; }}
    .badge-neutral {{ background: #6B728020; color: #6B7280; }}
    .badge-primary {{ background: var(--primary)20; color: var(--primary); }}

    /* Confidence Bar */
    .conf-bar-wrap {{
        background: var(--surface2);
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
        margin-top: 0.5rem;
    }}
    .conf-bar-fill {{
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    /* Wellness Tip */
    .wellness-box {{
        background: linear-gradient(135deg, var(--primary)10, var(--secondary)10);
        border-left: 4px solid var(--primary);
        border-radius: 0 16px 16px 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }}

    /* Crisis Alert */
    .crisis-critical {{ background: #7F1D1D20; border-left: 5px solid #7F1D1D; padding: 1rem; border-radius: 8px; }}
    .crisis-high {{ background: #EF444420; border-left: 5px solid #EF4444; padding: 1rem; border-radius: 8px; }}
    .crisis-moderate {{ background: #F59E0B20; border-left: 5px solid #F59E0B; padding: 1rem; border-radius: 8px; }}
    .crisis-low {{ background: #FBBF2420; border-left: 5px solid #FBBF24; padding: 1rem; border-radius: 8px; }}

    /* Profile Image */
    .profile-circle {{
        width: 80px; height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid var(--primary);
        box-shadow: 0 0 0 4px var(--primary)20;
    }}

    /* Achievement Badge */
    .achievement {{
        background: linear-gradient(135deg, var(--accent)20, var(--accent)08);
        border: 1px solid var(--accent)40;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
    }}
    .achievement .ach-icon {{ font-size: 2rem; }}
    .achievement .ach-name {{ font-weight: 700; font-size: 0.9rem; color: var(--text); }}
    .achievement .ach-desc {{ font-size: 0.75rem; color: var(--text-secondary); }}

    /* Streak Badge */
    .streak-badge {{
        background: linear-gradient(135deg, #F97316, #EF4444);
        color: white;
        border-radius: 999px;
        padding: 0.4rem 1rem;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }}

    /* Quote Card */
    .quote-card {{
        background: linear-gradient(135deg, var(--primary)15, var(--secondary)10);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid var(--primary)20;
        position: relative;
    }}
    .quote-card::before {{
        content: '"';
        font-size: 5rem;
        color: var(--primary)30;
        position: absolute;
        top: -0.5rem;
        left: 1rem;
        font-family: serif;
        line-height: 1;
    }}

    /* Navigation styling */
    .nav-item {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.65rem 1rem;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.15s ease;
        color: var(--text-secondary);
        text-decoration: none;
        font-weight: 500;
    }}
    .nav-item:hover, .nav-item.active {{
        background: var(--primary)12;
        color: var(--primary);
    }}

    /* Toggle switch */
    .stCheckbox > label {{
        color: var(--text) !important;
    }}

    /* Streamlit element overrides */
    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {{
        background: var(--surface) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
    }}
    .stTextInput input, .stTextArea textarea, .stSelectbox select {{
        color: var(--text) !important;
        background: var(--surface) !important;
    }}
    .stButton > button {{
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 20px var(--shadow) !important;
    }}
    [data-testid="stMetric"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: var(--primary) !important;
        font-weight: 800 !important;
    }}
    [data-testid="stExpander"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--surface2) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: var(--surface) !important;
        color: var(--primary) !important;
        font-weight: 700 !important;
    }}
    .stProgress > div > div {{
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        border-radius: 999px !important;
    }}
    .stProgress > div {{
        background: var(--surface2) !important;
        border-radius: 999px !important;
    }}
    div[data-testid="stAlert"] {{
        border-radius: 12px !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text) !important;
    }}
    p, span, li {{
        color: var(--text) !important;
    }}
    label {{
        color: var(--text) !important;
    }}
    .stRadio label, .stSelectbox label {{
        color: var(--text) !important;
    }}
    hr {{
        border-color: var(--border) !important;
    }}

    /* Animations */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes float {{
        0%,100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}
    @keyframes pulse-ring {{
        0% {{ transform: scale(0.9); opacity: 1; }}
        100% {{ transform: scale(1.4); opacity: 0; }}
    }}
    @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    @keyframes gradient-shift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .fade-in {{ animation: fadeInUp 0.4s ease both; }}
    .float {{ animation: float 3s ease-in-out infinite; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--surface2); border-radius: 999px; }}
    ::-webkit-scrollbar-thumb {{ background: var(--primary)50; border-radius: 999px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--primary); }}

    /* Password strength */
    .strength-bar {{
        height: 4px;
        border-radius: 999px;
        transition: all 0.3s ease;
        background: var(--border);
    }}

    /* Login page gradient */
    .login-hero {{
        background: linear-gradient(-45deg, #6366F1, #8B5CF6, #EC4899, #10B981);
        background-size: 400% 400%;
        animation: gradient-shift 8s ease infinite;
    }}

    /* Sidebar profile area */
    .sidebar-profile {{
        background: linear-gradient(135deg, var(--primary)15, var(--secondary)10);
        border-radius: 20px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid var(--primary)20;
        margin-bottom: 1rem;
    }}
    .sidebar-username {{
        font-weight: 700;
        font-size: 1rem;
        color: var(--text);
        margin: 0.5rem 0 0.2rem 0;
    }}
    .sidebar-role {{
        font-size: 0.75rem;
        color: var(--text-secondary);
    }}

    /* Notification dot */
    .notif-dot {{
        width: 8px; height: 8px;
        background: var(--danger);
        border-radius: 50%;
        display: inline-block;
        margin-left: 4px;
        animation: pulse-ring 1.5s ease-out infinite;
    }}

    /* Voice recorder */
    .voice-recorder {{
        border: 2px dashed var(--primary)50;
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        background: linear-gradient(135deg, var(--primary)05, var(--secondary)05);
        transition: all 0.2s ease;
    }}
    .voice-recorder:hover {{
        border-color: var(--primary);
        background: linear-gradient(135deg, var(--primary)10, var(--secondary)08);
    }}

    /* Ripple button */
    .ripple-btn {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .ripple-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--primary)40;
    }}

    /* Hide Streamlit branding */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
</style>
"""

st.markdown(get_theme_css(), unsafe_allow_html=True)

# ==================== USER DB ====================
USER_DB_FILE = 'users.json'
PROFILE_IMAGES_DIR = 'profile_images'
if not os.path.exists(PROFILE_IMAGES_DIR):
    os.makedirs(PROFILE_IMAGES_DIR)

def init_user_db():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'w') as f:
            json.dump({}, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_profile_image(username, image_data):
    try:
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
    try:
        images = list(Path(PROFILE_IMAGES_DIR).glob(f"{username}_*"))
        if images:
            return str(images[-1])
    except:
        pass
    return None

def create_user(username, password, email):
    try:
        users = {}
        if os.path.exists(USER_DB_FILE):
            with open(USER_DB_FILE, 'r') as f:
                users = json.load(f)
        if username in users:
            return False, "Username already exists!"
        users[username] = {
            'password': hash_password(password),
            'email': email,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'history': [],
            'settings': {'notifications': True, 'save_history': True, 'theme': 'Light', 'font_size': 'medium'},
            'streak': 0, 'last_analysis_date': None, 'achievements': []
        }
        with open(USER_DB_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        return True, "Account created successfully!"
    except Exception as e:
        return False, f"Error creating account: {str(e)}"

def verify_user(username, password):
    try:
        if not os.path.exists(USER_DB_FILE):
            return False, "No users found. Please sign up first."
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        if username not in users:
            return False, "Invalid username or password"
        if hash_password(password) == users[username]['password']:
            return True, users[username]
        return False, "Invalid username or password"
    except Exception as e:
        return False, f"Error: {str(e)}"

def save_analysis_to_history(username, analysis_data):
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        if username in users:
            if 'history' not in users[username]:
                users[username]['history'] = []
            if 'timestamp' not in analysis_data:
                analysis_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            users[username]['history'].append({'timestamp': analysis_data['timestamp'], 'data': analysis_data})
            if len(users[username]['history']) > 50:
                users[username]['history'] = users[username]['history'][-50:]
            with open(USER_DB_FILE, 'w') as f:
                json.dump(users, f, indent=4)
            return True
    except Exception as e:
        print(f"Error saving to history: {e}")
    return False

def load_user_history(username):
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        if username in users and 'history' in users[username]:
            return users[username]['history']
    except:
        pass
    return []

def update_user_settings(username, settings):
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        if username in users:
            users[username]['settings'] = settings
            with open(USER_DB_FILE, 'w') as f:
                json.dump(users, f, indent=4)
            return True
    except:
        pass
    return False

init_user_db()

# ==================== STREAK & ACHIEVEMENTS ====================
def update_streak():
    today = datetime.now().strftime("%Y-%m-%d")
    last = st.session_state.get('last_analysis_date')
    if last:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if last == today:
            pass
        elif last == yesterday:
            st.session_state.streak = st.session_state.get('streak', 0) + 1
            st.session_state.last_analysis_date = today
        else:
            st.session_state.streak = 1
            st.session_state.last_analysis_date = today
    else:
        st.session_state.streak = 1
        st.session_state.last_analysis_date = today

def check_achievements():
    history = st.session_state.analysis_history
    total = len(history)
    voice_count = sum(1 for e in history if e.get('type') == 'voice_analysis')
    text_count = sum(1 for e in history if e.get('type') == 'text_analysis')
    streak = st.session_state.get('streak', 0)
    earned = set(st.session_state.get('achievements', []))
    if total >= 1: earned.add('first_analysis')
    if total >= 10: earned.add('ten_analyses')
    if voice_count >= 5: earned.add('voice_master')
    if text_count >= 5: earned.add('journal_pro')
    if streak >= 3: earned.add('streak_3')
    if streak >= 7: earned.add('streak_7')
    st.session_state.achievements = list(earned)

def get_daily_quote():
    day_idx = datetime.now().timetuple().tm_yday % len(DAILY_QUOTES)
    return DAILY_QUOTES[day_idx]

def get_wellness_score():
    history = st.session_state.analysis_history
    if not history:
        return 50, "neutral"
    recent = history[-10:]
    positive = sum(1 for e in recent if MENTAL_HEALTH_MAPPING.get(e.get('dominant_emotion', ''), {}).get('category') == 'Positive')
    negative = sum(1 for e in recent if MENTAL_HEALTH_MAPPING.get(e.get('dominant_emotion', ''), {}).get('category') == 'Negative')
    score = max(0, min(100, 50 + (positive - negative) * 5))
    if score >= 70: trend = "improving"
    elif score >= 40: trend = "stable"
    else: trend = "needs_attention"
    return score, trend

# ==================== MODEL LOADING ====================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_emotion_model():
    model_path = 'emotiondetector.h5'
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except:
            return None
    return None

model = load_emotion_model()

# ==================== VOICE ANALYZER ====================
class EnhancedVoiceEmotionAnalyzer:
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 20
        self.frame_length = 2048
        self.hop_length = 512
        self.emotion_thresholds = {
            'happy': {'pitch_mean': (180, 320), 'pitch_std': (40, 100), 'energy_mean': (0.08, 0.25), 'energy_std': (0.03, 0.15), 'speech_rate': (140, 220), 'spectral_centroid': (2000, 3500), 'zcr_mean': (0.05, 0.15)},
            'sad': {'pitch_mean': (80, 160), 'pitch_std': (15, 40), 'energy_mean': (0.01, 0.08), 'energy_std': (0.005, 0.03), 'speech_rate': (70, 120), 'spectral_centroid': (800, 1800), 'zcr_mean': (0.02, 0.08)},
            'angry': {'pitch_mean': (200, 400), 'pitch_std': (60, 150), 'energy_mean': (0.15, 0.4), 'energy_std': (0.08, 0.25), 'speech_rate': (160, 250), 'spectral_centroid': (2500, 4500), 'zcr_mean': (0.1, 0.25)},
            'fearful': {'pitch_mean': (180, 350), 'pitch_std': (50, 120), 'energy_mean': (0.03, 0.15), 'energy_std': (0.02, 0.1), 'speech_rate': (120, 190), 'spectral_centroid': (1800, 3000), 'zcr_mean': (0.06, 0.18)},
            'calm': {'pitch_mean': (100, 180), 'pitch_std': (20, 50), 'energy_mean': (0.01, 0.06), 'energy_std': (0.005, 0.02), 'speech_rate': (80, 130), 'spectral_centroid': (1000, 2000), 'zcr_mean': (0.02, 0.06)},
            'surprised': {'pitch_mean': (250, 450), 'pitch_std': (80, 180), 'energy_mean': (0.1, 0.3), 'energy_std': (0.05, 0.2), 'speech_rate': (150, 230), 'spectral_centroid': (2800, 5000), 'zcr_mean': (0.08, 0.22)},
            'neutral': {'pitch_mean': (120, 200), 'pitch_std': (25, 60), 'energy_mean': (0.02, 0.1), 'energy_std': (0.01, 0.05), 'speech_rate': (100, 150), 'spectral_centroid': (1200, 2200), 'zcr_mean': (0.03, 0.1)}
        }
        self.feature_weights = {'pitch_mean': 0.15, 'pitch_std': 0.10, 'energy_mean': 0.12, 'energy_std': 0.08, 'speech_rate': 0.10, 'spectral_centroid': 0.08, 'zcr_mean': 0.07, 'mfcc': 0.30}

    def extract_enhanced_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30)
            if len(y) < self.sample_rate:
                return None, None, None
            features = {}
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
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
                features.update({'pitch_mean': 150.0, 'pitch_std': 50.0, 'pitch_max': 200.0, 'pitch_min': 100.0, 'pitch_range': 100.0, 'pitch_variability': 0.33})
            rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            features['energy_max'] = float(np.max(rms))
            features['energy_min'] = float(np.min(rms))
            features['energy_range'] = float(features['energy_max'] - features['energy_min'])
            features['energy_variability'] = float(features['energy_std'] / features['energy_mean'] if features['energy_mean'] > 0 else 0)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.frame_length, hop_length=self.hop_length)
            for i in range(self.n_mfcc):
                features[f'mfcc_mean_{i+1}'] = float(np.mean(mfccs[i]))
                features[f'mfcc_std_{i+1}'] = float(np.std(mfccs[i]))
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            harmonic, percussive = librosa.effects.hpss(y)
            features['harmonic_ratio'] = float(np.sum(np.abs(harmonic)) / (np.sum(np.abs(percussive)) + 1e-10))
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            features['tempo'] = float(tempo)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if len(onset_frames) > 1:
                features['speech_rate'] = float(len(onset_frames) / (len(y) / sr) * 60)
            else:
                features['speech_rate'] = 120.0
            non_silent = librosa.effects.split(y, top_db=30)
            if len(non_silent) > 0:
                total_speech = np.sum([end - start for start, end in non_silent])
                silence_duration = len(y) - total_speech
                features['speech_percentage'] = float(total_speech / len(y) * 100)
                features['pause_count'] = len(non_silent) - 1
                features['avg_pause_duration'] = float(silence_duration / max(features['pause_count'], 1))
            else:
                features.update({'speech_percentage': 100.0, 'pause_count': 0, 'avg_pause_duration': 0.0})
            return features, y, sr
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

    def detect_emotion_ml(self, features):
        if not features:
            return 'neutral', 0.5, {e: 1.0/len(VOICE_EMOTION_MAPPING) for e in VOICE_EMOTION_MAPPING}
        emotion_scores = {e: 0.0 for e in VOICE_EMOTION_MAPPING}
        for emotion, thresholds in self.emotion_thresholds.items():
            score = 0.0; weight_sum = 0.0
            for feature, (low, high) in thresholds.items():
                if feature in features:
                    value = features[feature]
                    if low <= value <= high: feature_score = 1.0
                    elif value < low: feature_score = max(0, 1 - (low - value) / low)
                    else: feature_score = max(0, 1 - (value - high) / high)
                    weight = self.feature_weights.get(feature, 0.05)
                    score += feature_score * weight; weight_sum += weight
            emotion_scores[emotion] = score / weight_sum if weight_sum > 0 else 0.5
        mfcc_features = [features.get(f'mfcc_mean_{i}', 0) for i in range(1, 13)]
        if np.mean(mfcc_features[:4]) > -50 and np.std(mfcc_features[:4]) > 30: emotion_scores['happy'] += 0.15
        elif np.mean(mfcc_features[:4]) < -100: emotion_scores['sad'] += 0.15
        elif np.mean(mfcc_features[4:8]) > 50: emotion_scores['angry'] += 0.15
        elif np.std(mfcc_features) > 60: emotion_scores['fearful'] += 0.15
        elif np.std(mfcc_features) < 30: emotion_scores['calm'] += 0.15
        else: emotion_scores['neutral'] += 0.15
        total = sum(emotion_scores.values())
        if total > 0:
            for e in emotion_scores: emotion_scores[e] /= total
        exp_scores = np.exp([emotion_scores[e] for e in emotion_scores])
        exp_scores = exp_scores / np.sum(exp_scores)
        for i, e in enumerate(emotion_scores): emotion_scores[e] = float(exp_scores[i])
        dominant = max(emotion_scores, key=emotion_scores.get)
        return dominant, emotion_scores[dominant], emotion_scores

    def analyze_voice_health_advanced(self, features):
        health = []
        if not features: return health
        if features.get('pitch_mean', 0) > 300 and features.get('energy_mean', 0) > 0.15:
            sev = 'high' if features.get('zcr_mean', 0) > 0.1 else 'moderate'
            health.append({'issue': 'Vocal strain detected', 'suggestion': 'Rest your voice, stay hydrated.', 'severity': sev, 'clinical_term': 'Vocal Hyperfunction', 'exercises': ['Lip trills', 'Humming', 'Semi-occluded exercises']})
        if features.get('energy_mean', 0) < 0.03 and features.get('zcr_mean', 0) > 0.12:
            health.append({'issue': 'Breathy voice quality', 'suggestion': 'Practice sustained vowel sounds. Consult ENT if persistent.', 'severity': 'moderate', 'clinical_term': 'Breathy Dysphonia', 'exercises': ['Sustained "ah" vowel', 'Resonant voice therapy']})
        if features.get('pitch_std', 100) < 25:
            health.append({'issue': 'Reduced pitch variability', 'suggestion': 'Try reading aloud with exaggerated expression.', 'severity': 'moderate', 'clinical_term': 'Monopitch', 'exercises': ['Pitch glides', 'Singing exercises']})
        return health

voice_analyzer = EnhancedVoiceEmotionAnalyzer()

# ==================== TEXT ANALYZER ====================
class EnhancedTextEmotionAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.emotion_patterns = {
            'joy': {'words': {'happy': 0.8, 'joy': 1.0, 'delighted': 0.9, 'pleased': 0.7, 'grateful': 0.8, 'wonderful': 0.9, 'fantastic': 0.9, 'amazing': 0.8, 'love': 0.9, 'glad': 0.7, 'cheerful': 0.8, 'excited': 0.8, 'thrilled': 0.9, 'blessed': 0.7, 'great': 0.7, 'excellent': 0.8, 'bliss': 1.0, 'ecstatic': 1.0, 'elated': 0.9}, 'phrases': {'so happy': 0.9, 'made my day': 1.0, 'best day': 0.9, 'on top of the world': 1.0}, 'intensifiers': ['so', 'very', 'extremely', 'really', 'absolutely']},
            'sadness': {'words': {'sad': 0.8, 'depressed': 1.0, 'unhappy': 0.7, 'miserable': 0.9, 'hopeless': 1.0, 'devastated': 1.0, 'heartbroken': 1.0, 'grief': 1.0, 'sorrow': 0.9, 'despair': 1.0, 'lonely': 0.8, 'crying': 0.7, 'hurt': 0.7, 'suffering': 0.8, 'empty': 0.8, 'melancholy': 0.8}, 'phrases': {'so sad': 0.8, 'want to cry': 0.8, 'no hope': 0.9, 'broken heart': 0.9}},
            'anger': {'words': {'angry': 0.9, 'mad': 0.7, 'furious': 1.0, 'frustrated': 0.8, 'irritated': 0.7, 'outraged': 1.0, 'hate': 0.9, 'rage': 1.0, 'fury': 1.0, 'fuming': 0.9, 'livid': 0.9, 'enraged': 1.0}, 'phrases': {'fed up': 0.7, 'had enough': 0.8, 'pissed off': 0.8}},
            'fear': {'words': {'afraid': 0.8, 'scared': 0.8, 'terrified': 1.0, 'anxious': 0.8, 'worried': 0.7, 'nervous': 0.7, 'panicked': 0.9, 'frightened': 0.8, 'dread': 0.9, 'anxiety': 0.8, 'panic': 0.9, 'paranoid': 0.7, 'apprehensive': 0.7}, 'phrases': {'scared to death': 1.0, 'panic attack': 0.9, 'heart racing': 0.7}},
            'surprise': {'words': {'surprised': 0.7, 'shocked': 0.8, 'amazed': 0.8, 'astonished': 0.8, 'stunned': 0.8, 'unexpected': 0.7, 'incredible': 0.7, 'unbelievable': 0.8, 'wow': 0.7}, 'phrases': {"can't believe": 0.8, 'no way': 0.7, 'oh my god': 0.7}},
            'trust': {'words': {'trust': 0.8, 'confident': 0.7, 'believe': 0.6, 'faith': 0.7, 'reliable': 0.7, 'assured': 0.6, 'trustworthy': 0.8, 'honest': 0.7, 'loyal': 0.7}, 'phrases': {'i trust': 0.8, 'have faith': 0.7}},
            'anticipation': {'words': {'expect': 0.6, 'anticipate': 0.7, 'await': 0.6, 'hope': 0.7, 'hoping': 0.7, 'wish': 0.6, 'excited for': 0.8}, 'phrases': {'looking forward': 0.8, "can't wait": 0.8, 'hope for': 0.7}}
        }

    def analyze_sentiment(self, text):
        vader = self.sia.polarity_scores(text)
        blob = TextBlob(text)
        return {'compound': vader['compound'], 'positive': vader['pos'], 'negative': vader['neg'], 'neutral': vader['neu'], 'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}

    def detect_emotions(self, text):
        text_lower = text.lower()
        words = nltk.word_tokenize(text_lower)
        emotion_scores = {}
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            for word, weight in patterns['words'].items():
                count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
                if count > 0:
                    score += count * weight * 0.3
            for phrase, weight in patterns.get('phrases', {}).items():
                if phrase in text_lower:
                    score += weight * 0.5
            for intensifier in patterns.get('intensifiers', []):
                for word in patterns['words']:
                    if f"{intensifier} {word}" in text_lower:
                        score += 0.3
            word_count = len(words)
            if word_count > 0:
                score = score / (word_count ** 0.5)
            if score > 0:
                emotion_scores[emotion] = round(score, 3)
        if not emotion_scores:
            sentiment = self.analyze_sentiment(text)
            c = sentiment['compound']
            if c >= 0.5: emotion_scores = {'joy': 0.6, 'trust': 0.3, 'anticipation': 0.1}
            elif c <= -0.5: emotion_scores = {'sadness': 0.6, 'fear': 0.3, 'anger': 0.1}
            elif c > 0.1: emotion_scores = {'trust': 0.5, 'anticipation': 0.3, 'joy': 0.2}
            elif c < -0.1: emotion_scores = {'fear': 0.4, 'sadness': 0.4, 'anger': 0.2}
            else: emotion_scores = {'neutral': 1.0}
        total = sum(emotion_scores.values())
        if total > 0:
            for e in emotion_scores: emotion_scores[e] /= total
        return emotion_scores

    def assess_crisis_risk(self, text):
        text_lower = text.lower()
        crisis_indicators = {
            'critical': ['suicide', 'kill myself', 'end my life', 'want to die', 'self-harm', 'hurt myself', 'no reason to live', 'better off dead', 'take my life', 'end it all'],
            'high': ['hopeless', "can't go on", 'giving up', 'no way out', 'tired of living', 'worthless', 'pointless', 'nothing matters', 'no hope', "can't take it"],
            'moderate': ['depressed', 'sad', 'lonely', 'alone', 'struggling', 'anxious', 'scared', 'worried', 'stress', 'exhausted'],
            'low': ['down', 'blue', 'unhappy', 'frustrated', 'upset', 'disappointed']
        }
        risk_level = 'none'; indicators_found = []; severity_score = 0
        for level, indicators in crisis_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    indicators_found.append(indicator)
                    if level == 'critical': severity_score += 10; risk_level = 'CRITICAL'
                    elif level == 'high':
                        severity_score += 5
                        if risk_level not in ['CRITICAL']: risk_level = 'HIGH'
                    elif level == 'moderate':
                        severity_score += 2
                        if risk_level not in ['CRITICAL', 'HIGH']: risk_level = 'MODERATE'
                    elif level == 'low':
                        severity_score += 1
                        if risk_level not in ['CRITICAL', 'HIGH', 'MODERATE']: risk_level = 'LOW'
        return {'risk_level': risk_level, 'indicators': list(set(indicators_found)), 'severity_score': severity_score, 'requires_immediate_attention': risk_level in ['CRITICAL', 'HIGH'], 'indicator_count': len(set(indicators_found))}

    def analyze_text(self, text):
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        word_count = len(words); char_count = len(text); sentence_count = len(sentences)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        unique_words = len(set([w.lower() for w in words]))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        try:
            pos_tags = nltk.pos_tag(words)
            noun_count = len([t for _, t in pos_tags if t.startswith('N')])
            verb_count = len([t for _, t in pos_tags if t.startswith('V')])
            adj_count = len([t for _, t in pos_tags if t.startswith('J')])
            adv_count = len([t for _, t in pos_tags if t.startswith('R')])
            pronoun_count = len([t for _, t in pos_tags if t.startswith('PRP')])
        except:
            noun_count = verb_count = adj_count = adv_count = pronoun_count = 0
        sentiment = self.analyze_sentiment(text)
        emotions = self.detect_emotions(text)
        crisis = self.assess_crisis_risk(text)
        dominant = max(emotions, key=emotions.get) if emotions else 'neutral'
        wellness = TEXT_EMOTION_MAPPING.get(dominant, {'status': '😐 Neutral', 'suggestion': 'Balanced state.', 'activities': ['Mindfulness', 'Read', 'Walk'], 'color': '#808080', 'wellness_tip': 'Balance is good.', 'icon': '😐', 'category': 'Neutral'})
        return {
            'dominant_emotion': dominant,
            'dominant_score': emotions.get(dominant, 0),
            'emotions_detected': emotions,
            'sentiment': sentiment,
            'crisis_risk': crisis,
            'wellness': wellness,
            'statistics': {'word_count': word_count, 'char_count': char_count, 'sentence_count': sentence_count, 'avg_word_length': round(avg_word_length, 2), 'avg_sentence_length': round(avg_sentence_length, 2), 'lexical_diversity': round(lexical_diversity, 3), 'unique_words': unique_words, 'noun_count': noun_count, 'verb_count': verb_count, 'adj_count': adj_count, 'adv_count': adv_count, 'pronoun_count': pronoun_count},
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'full_text': text
        }

text_analyzer = EnhancedTextEmotionAnalyzer()

# ==================== PDF GENERATOR ====================
class ProfessionalPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_fill_color(99, 102, 241)
        self.rect(0, 0, 210, 40, 'F')
        self.set_y(10)
        self.set_font('Helvetica', 'B', 22)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'NEUROWELL MENTAL HEALTH CENTER', 0, 1, 'C')
        self.set_font('Helvetica', '', 11)
        self.set_text_color(200, 200, 255)
        self.cell(0, 8, 'Emotional Wellness Assessment Report', 0, 1, 'C')
        self.set_y(44)
        self.set_draw_color(16, 185, 129)
        self.set_line_width(1)
        self.line(10, 44, 200, 44)
        self.ln(12)

    def footer(self):
        self.set_y(-20)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f'Page {self.page_no()} | NeuroWell AI System | Confidential', 0, 0, 'C')

    def _clean(self, text):
        if not text: return ""
        text = str(text)
        replacements = {'•': '-', '●': '-', '→': '->', '←': '<-', '—': '-', '–': '-', '…': '...', '"': '"', '"': '"', ''': "'", ''': "'"}
        for k, v in replacements.items(): text = text.replace(k, v)
        text = re.sub(r'[^\x00-\x7F]', '', text)
        return text

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(99, 102, 241)
        self.cell(0, 10, self._clean(title), 0, 1)
        self.set_draw_color(99, 102, 241)
        self.line(self.get_x(), self.get_y(), 200, self.get_y())
        self.ln(6)

    def create_table(self, headers, data, col_widths):
        self.set_fill_color(99, 102, 241)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 9)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 9, self._clean(header), 1, 0, 'C', 1)
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font('Helvetica', '', 9)
        fill = False
        for row in data:
            if fill: self.set_fill_color(245, 245, 255)
            else: self.set_fill_color(255, 255, 255)
            for i, item in enumerate(row):
                self.cell(col_widths[i], 7, self._clean(str(item)), 'LR', 0, 'C', True)
            self.ln()
            fill = not fill
        self.cell(sum(col_widths), 0, '', 'T')

def generate_pdf_report(data, username, report_type="comprehensive"):
    pdf = ProfessionalPDF()
    pdf.add_page()
    now = datetime.now()

    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 12, f'NeuroWell Assessment — {username}', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(0, 7, f'Generated: {now.strftime("%B %d, %Y at %H:%M")}', 0, 1, 'C')
    pdf.ln(8)

    facial = [e for e in data if e.get('type', 'facial_analysis') == 'facial_analysis']
    text_d = [e for e in data if e.get('type') == 'text_analysis']
    voice_d = [e for e in data if e.get('type') == 'voice_analysis']

    pdf.section_title('ANALYSIS SUMMARY')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(31, 41, 55)
    rows = [
        ('Facial Analyses', str(len(facial))),
        ('Text Analyses', str(len(text_d))),
        ('Voice Analyses', str(len(voice_d))),
        ('Total Sessions', str(len(data))),
    ]
    for label, val in rows:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(70, 7, pdf._clean(label + ':'), 0, 0)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 7, val, 0, 1)
    pdf.ln(8)

    all_emotions = [e.get('dominant_emotion', 'neutral') for e in data]
    if all_emotions:
        pdf.section_title('EMOTION DISTRIBUTION')
        counts = Counter(all_emotions)
        total = len(all_emotions)
        headers = ['Emotion', 'Count', 'Percentage']
        col_widths = [60, 40, 60]
        table_data = [[e.capitalize(), str(c), f'{c/total*100:.1f}%'] for e, c in sorted(counts.items())]
        pdf.create_table(headers, table_data, col_widths)
        pdf.ln(10)

    if text_d:
        pdf.section_title('TEXT ANALYSIS HIGHLIGHTS')
        crises = sum(1 for e in text_d if e.get('crisis_risk', {}).get('risk_level', 'none') != 'none')
        avg_sent = sum(e.get('sentiment', {}).get('compound', 0) for e in text_d) / len(text_d)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 7, f'Average Sentiment Score: {avg_sent:.2f}', 0, 1)
        pdf.cell(0, 7, f'Crisis Alerts Detected: {crises}', 0, 1)
        pdf.ln(8)

    pdf.section_title('WELLNESS RECOMMENDATIONS')
    most_common_emotion = Counter(all_emotions).most_common(1)[0][0] if all_emotions else 'neutral'
    mhd = MENTAL_HEALTH_MAPPING.get(most_common_emotion, MENTAL_HEALTH_MAPPING['neutral'])
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, pdf._clean(f"Primary State: {most_common_emotion.capitalize()} — {mhd['suggestion']}"))
    pdf.ln(3)
    for act in mhd['activities']:
        pdf.cell(0, 6, pdf._clean(f"• {act}"), 0, 1)

    pdf.set_y(-50)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 4, 'CONFIDENTIALITY NOTICE: This report contains protected health information. Unauthorized distribution is prohibited.', 0, 'C')

    return pdf.output(dest='S').encode('latin-1', errors='ignore')

# ==================== UI HELPERS ====================
def get_profile_img_html(username, size=80, css_class="profile-circle"):
    path = st.session_state.get('profile_image') or get_profile_image(username)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" style="border-radius:50%;object-fit:cover;border:3px solid var(--primary);box-shadow:0 0 0 4px var(--primary)20;" />'
    return f'<img src="https://api.dicebear.com/7.x/initials/svg?seed={username}&backgroundColor=6366f1&textColor=ffffff" width="{size}" height="{size}" style="border-radius:50%;border:3px solid var(--primary);" />'

def render_emotion_card(emotion, confidence, color, show_bar=True):
    C = get_colors()
    badge_class = 'badge-positive' if MENTAL_HEALTH_MAPPING.get(emotion, {}).get('category') == 'Positive' else 'badge-negative' if MENTAL_HEALTH_MAPPING.get(emotion, {}).get('category') == 'Negative' else 'badge-neutral'
    bar_html = f'<div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{confidence*100:.0f}%"></div></div>' if show_bar else ''
    return f"""
    <div class="nw-card" style="border-top: 4px solid {color};">
        <div style="display:flex;align-items:center;gap:1rem;">
            <div style="font-size:3rem;">{MENTAL_HEALTH_MAPPING.get(emotion,{}).get('icon','😐')}</div>
            <div style="flex:1;">
                <div style="font-size:1.4rem;font-weight:800;color:{color};">{emotion.upper()}</div>
                <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem;">
                    <span class="badge {badge_class}">{MENTAL_HEALTH_MAPPING.get(emotion,{}).get('category','Neutral')}</span>
                    <span style="color:var(--text-secondary);font-size:0.85rem;">{confidence*100:.1f}% confidence</span>
                </div>
                {bar_html}
            </div>
        </div>
    </div>
    """

def render_metric_pill(value, label, color=None):
    color = color or 'var(--primary)'
    return f"""<div class="metric-pill"><span class="value" style="color:{color};">{value}</span><div class="label">{label}</div></div>"""

# ==================== LOGIN PAGE ====================
def show_login_signup():
    C = get_colors()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class="login-hero" style="min-height:100vh;padding:3rem 2.5rem;border-radius:0 32px 32px 0;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;">
            <div class="float" style="margin-bottom:2rem;">
                <div style="font-size:5rem;">🧠</div>
            </div>
            <h1 style="color:white;font-size:3rem;font-weight:900;margin:0;text-shadow:0 2px 20px rgba(0,0,0,0.3);">NeuroWell</h1>
            <p style="color:rgba(255,255,255,0.9);font-size:1.2rem;margin:0.75rem 0 2rem 0;">Your AI Mental Health Companion</p>
            <div style="display:grid;gap:1rem;width:100%;max-width:340px;">
                <div style="background:rgba(255,255,255,0.15);border-radius:16px;padding:1.25rem;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.2);">
                    <div style="font-size:1.5rem;">🎭</div>
                    <div style="color:white;font-weight:600;margin-top:0.5rem;">Facial Analysis</div>
                    <div style="color:rgba(255,255,255,0.75);font-size:0.85rem;">Detect emotions from expressions</div>
                </div>
                <div style="background:rgba(255,255,255,0.15);border-radius:16px;padding:1.25rem;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.2);">
                    <div style="font-size:1.5rem;">📝</div>
                    <div style="color:white;font-weight:600;margin-top:0.5rem;">Text & Voice Analysis</div>
                    <div style="color:rgba(255,255,255,0.75);font-size:0.85rem;">NLP + ML-based emotion detection</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='padding:3rem 2rem;'>", unsafe_allow_html=True)

        # Theme toggle at top
        col_a, col_b = st.columns([3, 1])
        with col_b:
            dm = st.toggle("🌙", value=st.session_state.dark_mode, key="login_dm")
            if dm != st.session_state.dark_mode:
                st.session_state.dark_mode = dm
                st.rerun()

        st.markdown(f"""
        <h2 style="font-weight:800;font-size:2rem;margin-bottom:0.25rem;color:var(--text);">Welcome back 👋</h2>
        <p style="color:var(--text-secondary);margin-bottom:2rem;">Sign in to continue your wellness journey</p>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab1:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                col_a, col_b = st.columns(2)
                with col_a:
                    remember = st.checkbox("Remember me")
                submitted = st.form_submit_button("Sign In →", use_container_width=True, type="primary")
                if submitted:
                    if username and password:
                        with st.spinner("Authenticating..."):
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
                                check_achievements()
                                st.success("✅ Welcome back!")
                                st.rerun()
                            else:
                                st.error(f"❌ {result}")
                    else:
                        st.warning("Please fill in all fields")

            st.markdown("""
            <div style="text-align:center;margin-top:1.5rem;">
                <p style="color:var(--text-secondary);font-size:0.85rem;">Demo: Use any username/password or create an account</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            with st.form("signup_form", clear_on_submit=False):
                new_username = st.text_input("Username", placeholder="Choose a username", key="su_user")
                new_email = st.text_input("Email", placeholder="your@email.com", key="su_email")
                new_password = st.text_input("Password", type="password", placeholder="Min. 6 characters", key="su_pass")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="su_conf")

                if new_password:
                    strength = 0
                    if len(new_password) >= 8: strength += 1
                    if any(c.isupper() for c in new_password): strength += 1
                    if any(c.isdigit() for c in new_password): strength += 1
                    if any(c in '!@#$%^&*' for c in new_password): strength += 1
                    colors = ['#EF4444', '#F97316', '#F59E0B', '#22C55E']
                    labels = ['Weak', 'Fair', 'Good', 'Strong']
                    pct = (strength / 4) * 100
                    st.markdown(f"""
                    <div style="margin:-0.5rem 0 0.5rem 0;">
                        <div class="conf-bar-wrap" style="height:4px;">
                            <div style="height:100%;width:{pct}%;background:{colors[min(strength-1,3)] if strength > 0 else '#E5E7EB'};border-radius:999px;transition:all 0.3s;"></div>
                        </div>
                        <p style="font-size:0.75rem;color:{colors[min(strength-1,3)] if strength>0 else 'var(--text-secondary)'};margin:0.25rem 0 0 0;">{labels[min(strength-1,3)] if strength > 0 else 'Enter password'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
                submitted2 = st.form_submit_button("Create Account →", use_container_width=True, type="primary")
                if submitted2:
                    if all([new_username, new_email, new_password, confirm_password]):
                        if new_password != confirm_password:
                            st.error("❌ Passwords don't match")
                        elif len(new_password) < 6:
                            st.error("❌ Password must be at least 6 characters")
                        elif not agree:
                            st.error("❌ Please agree to Terms of Service")
                        else:
                            with st.spinner("Creating your account..."):
                                success, message = create_user(new_username, new_password, new_email)
                                if success:
                                    st.success(f"✅ {message} Please sign in!")
                                    st.balloons()
                                else:
                                    st.error(f"❌ {message}")
                    else:
                        st.warning("Please fill in all fields")

        st.markdown("</div>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
def show_sidebar():
    C = get_colors()
    with st.sidebar:
        # Profile section
        profile_html = get_profile_img_html(st.session_state.username, 70)
        wellness_score, trend = get_wellness_score()
        trend_icon = "📈" if trend == "improving" else "⚡" if trend == "stable" else "💙"
        streak = st.session_state.get('streak', 0)

        st.markdown(f"""
        <div class="sidebar-profile fade-in">
            {profile_html}
            <div class="sidebar-username">{st.session_state.username}</div>
            <div class="sidebar-role">NeuroWell Member</div>
            <div style="display:flex;justify-content:center;gap:0.5rem;margin-top:0.75rem;flex-wrap:wrap;">
                <span class="streak-badge">🔥 {streak} day streak</span>
                <span class="badge badge-primary">{trend_icon} {trend.replace('_',' ').title()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Wellness score
        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                <span style="font-size:0.8rem;color:var(--text-secondary);font-weight:600;">WELLNESS SCORE</span>
                <span style="font-size:0.9rem;font-weight:700;color:var(--primary);">{wellness_score}/100</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{wellness_score}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        pages = [
            ("🏠", "Home"), ("🎭", "Facial Analysis"), ("📝", "Text Analysis"),
            ("🎤", "Voice Analysis"), ("💚", "Wellness Tips"), ("📋", "History"),
            ("📈", "Analytics"), ("👤", "Profile")
        ]
        for icon, name in pages:
            is_active = st.session_state.current_page == name
            if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True,
                        type="primary" if is_active else "secondary"):
                st.session_state.current_page = name
                st.rerun()

        st.markdown("---")

        # Theme & Settings
        col_a, col_b = st.columns(2)
        with col_a:
            dm = st.toggle("🌙 Dark", value=st.session_state.dark_mode, key="sidebar_dm")
            if dm != st.session_state.dark_mode:
                st.session_state.dark_mode = dm
                st.rerun()
        with col_b:
            fs = st.selectbox("Font", ["small", "medium", "large"], index=["small","medium","large"].index(st.session_state.font_size), label_visibility="collapsed", key="font_sel")
            if fs != st.session_state.font_size:
                st.session_state.font_size = fs
                st.rerun()

        # Quick stats
        if st.session_state.analysis_history:
            total = len(st.session_state.analysis_history)
            st.markdown(f"""
            <div style="background:var(--surface2);border-radius:12px;padding:0.75rem;margin-top:0.5rem;">
                <div style="font-size:0.7rem;color:var(--text-secondary);font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">Quick Stats</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;text-align:center;">
                    <div><div style="font-weight:700;color:var(--primary);">{total}</div><div style="font-size:0.7rem;color:var(--text-secondary);">Total</div></div>
                    <div><div style="font-weight:700;color:var(--secondary);">{len(st.session_state.get('achievements',[]))}</div><div style="font-size:0.7rem;color:var(--text-secondary);">Badges</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session()
            st.rerun()

# ==================== HOME PAGE ====================
def show_home():
    C = get_colors()
    username = st.session_state.username
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    quote_text, quote_author = get_daily_quote()
    wellness_score, trend = get_wellness_score()
    streak = st.session_state.get('streak', 0)

    # Hero header
    st.markdown(f"""
    <div class="fade-in" style="margin-bottom:2rem;">
        <h1 style="font-size:2.5rem;font-weight:900;margin:0;">{greeting}, {username} 👋</h1>
        <p style="color:var(--text-secondary);font-size:1.1rem;margin:0.5rem 0 0 0;">How are you feeling today? Let's check in.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top row: Wellness + Quote
    col1, col2 = st.columns([1, 2])
    with col1:
        score_color = '#22C55E' if wellness_score >= 70 else '#F59E0B' if wellness_score >= 40 else '#EF4444'
        st.markdown(f"""
        <div class="nw-card" style="text-align:center;border-top:4px solid {score_color};">
            <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-secondary);margin-bottom:0.75rem;">Wellness Score</div>
            <div style="position:relative;width:120px;height:120px;margin:0 auto 1rem auto;">
                <svg viewBox="0 0 120 120" style="transform:rotate(-90deg);">
                    <circle cx="60" cy="60" r="50" fill="none" stroke="var(--surface2)" stroke-width="10"/>
                    <circle cx="60" cy="60" r="50" fill="none" stroke="{score_color}" stroke-width="10"
                        stroke-dasharray="{2*3.14159*50}" stroke-dashoffset="{2*3.14159*50*(1-wellness_score/100)}"
                        stroke-linecap="round" style="transition:stroke-dashoffset 1s ease;"/>
                </svg>
                <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
                    <div style="font-size:1.8rem;font-weight:900;color:{score_color};">{wellness_score}</div>
                    <div style="font-size:0.65rem;color:var(--text-secondary);">/ 100</div>
                </div>
            </div>
            <div style="display:inline-flex;align-items:center;gap:0.4rem;background:{score_color}15;color:{score_color};border-radius:999px;padding:0.3rem 0.85rem;font-size:0.8rem;font-weight:600;">
                {'📈' if trend=='improving' else '⚡' if trend=='stable' else '💙'} {trend.replace('_',' ').title()}
            </div>
            <div style="margin-top:1rem;">
                <span class="streak-badge">🔥 {streak} day streak</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="quote-card fade-in" style="height:100%;box-sizing:border-box;">
            <div style="padding-top:1rem;">
                <p style="font-size:1.1rem;font-style:italic;color:var(--text);line-height:1.7;margin:0 0 1rem 0;">{quote_text}</p>
                <p style="font-weight:700;color:var(--primary);margin:0;">— {quote_author}</p>
                <div style="margin-top:1rem;font-size:0.75rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.05em;">Daily Inspiration • {datetime.now().strftime('%B %d')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick action cards
    st.markdown("<h3 style='font-weight:700;margin-bottom:1rem;'>Start Analysis</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    actions = [
        ("🎭", "Facial Analysis", "Analyze emotions from your face using AI", "Facial Analysis", "#6366F1"),
        ("📝", "Text Analysis", "Detect emotions in your writing or journal", "Text Analysis", "#10B981"),
        ("🎤", "Voice Analysis", "Understand your emotional state from speech", "Voice Analysis", "#F59E0B"),
    ]
    for col, (icon, title, desc, page, color) in zip([col1, col2, col3], actions):
        with col:
            st.markdown(f"""
            <div class="action-card" style="border-top:4px solid {color};">
                <span class="icon">{icon}</span>
                <div class="title">{title}</div>
                <div class="desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {title.split()[0]}", key=f"home_{page}", use_container_width=True, type="primary"):
                st.session_state.current_page = page
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Achievements & Recent Activity
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<h3 style='font-weight:700;margin-bottom:1rem;'>🏆 Achievements</h3>", unsafe_allow_html=True)
        earned = set(st.session_state.get('achievements', []))
        for ach_id, ach in ACHIEVEMENTS.items():
            unlocked = ach_id in earned
            opacity = "1" if unlocked else "0.3"
            st.markdown(f"""
            <div class="achievement" style="opacity:{opacity};margin-bottom:0.5rem;">
                <div class="ach-icon">{ach['icon']}</div>
                <div class="ach-name">{ach['name']}</div>
                <div class="ach-desc">{ach['desc']}</div>
                {'<div style="color:var(--success);font-size:0.7rem;font-weight:700;">✓ UNLOCKED</div>' if unlocked else ''}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='font-weight:700;margin-bottom:1rem;'>📋 Recent Activity</h3>", unsafe_allow_html=True)
        if st.session_state.analysis_history:
            recent = list(reversed(st.session_state.analysis_history[-5:]))
            for entry in recent:
                t = entry.get('type', 'facial_analysis')
                emotion = entry.get('dominant_emotion', 'neutral')
                ts = entry.get('timestamp', 'N/A')
                conf = entry.get('confidence', 0)
                color = MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#6B7280')
                type_icon = '🎭' if t == 'facial_analysis' else '📝' if t == 'text_analysis' else '🎤'
                st.markdown(f"""
                <div class="nw-card-flat" style="display:flex;align-items:center;gap:1rem;">
                    <div style="font-size:1.5rem;">{type_icon}</div>
                    <div style="flex:1;">
                        <div style="font-weight:600;color:{color};">{emotion.capitalize()}</div>
                        <div style="font-size:0.8rem;color:var(--text-secondary);">{ts[:16] if len(ts) > 16 else ts}</div>
                    </div>
                    <div style="font-size:0.85rem;font-weight:700;color:{color};">{conf*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="nw-card-flat" style="text-align:center;padding:2rem;">
                <div style="font-size:2rem;">🌱</div>
                <p style="color:var(--text-secondary);">No analyses yet. Start your wellness journey!</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== FACIAL ANALYSIS ====================
def show_facial_analysis():
    C = get_colors()
    st.markdown("""
    <div class="fade-in">
        <h2 style="font-weight:800;margin-bottom:0.25rem;">🎭 Facial Emotion Analysis</h2>
        <p style="color:var(--text-secondary);margin-bottom:1.5rem;">AI-powered emotion detection from facial expressions</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.markdown("""
        <div class="nw-card" style="border-top:4px solid var(--warning);text-align:center;padding:2rem;">
            <div style="font-size:3rem;">⚠️</div>
            <h3>Model Not Found</h3>
            <p style="color:var(--text-secondary);">The facial emotion model (emotiondetector.h5) is not available.<br>Text and Voice analysis are fully operational.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    option = st.radio("Input Method", ["📷 Live Camera", "📁 Upload Image"], horizontal=True)

    source = None
    if option == "📷 Live Camera":
        st.markdown("""
        <div class="voice-recorder" style="margin-bottom:1rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📷</div>
            <p style="color:var(--text-secondary);">Position your face clearly and ensure good lighting</p>
        </div>
        """, unsafe_allow_html=True)
        source = st.camera_input("Take a snapshot")
    else:
        source = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if source:
        def process(src, is_upload=False):
            file_bytes = np.asarray(bytearray(src.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1 if is_upload else 1.3, minNeighbors=2 if is_upload else 5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                roi_gray = gray[y:y+h, x:x+w]
            else:
                if is_upload:
                    roi_gray = gray
                else:
                    return None, None, "No face detected. Please ensure good lighting."
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            final_input = np.reshape(roi_normalized, (1, 48, 48, 1))
            preds = model.predict(final_input, verbose=0)[0]
            return roi_resized, preds, None

        with st.spinner("🔍 Analyzing facial expressions..."):
            cropped, preds, error = process(source, is_upload=(option == "📁 Upload Image"))

        if error:
            st.error(error)
        else:
            dominant = CLASS_NAMES[np.argmax(preds)]
            max_conf = float(np.max(preds))
            mhd = MENTAL_HEALTH_MAPPING[dominant]
            color = mhd['color']

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div class="nw-card" style="text-align:center;border-top:4px solid {color};">
                    <div style="font-size:4rem;margin-bottom:0.5rem;">{mhd['icon']}</div>
                    <div style="font-size:1.8rem;font-weight:900;color:{color};">{dominant.upper()}</div>
                    <div style="margin:0.75rem 0;">
                        <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:0.35rem;">Confidence</div>
                        <div style="font-size:2rem;font-weight:800;color:{color};">{max_conf*100:.1f}%</div>
                        <div class="conf-bar-wrap" style="margin-top:0.5rem;">
                            <div class="conf-bar-fill" style="width:{max_conf*100:.0f}%;background:{color};"></div>
                        </div>
                    </div>
                    <span class="badge {'badge-positive' if mhd['category']=='Positive' else 'badge-negative' if mhd['category']=='Negative' else 'badge-neutral'}">{mhd['category']}</span>
                </div>
                """, unsafe_allow_html=True)

                if cropped is not None:
                    st.image(cropped, caption="Analyzed region", width=200)

            with col2:
                st.markdown(f"""
                <div class="wellness-box">
                    <h4 style="color:var(--primary);margin:0 0 0.5rem 0;">{mhd['status']}</h4>
                    <p style="margin:0 0 0.5rem 0;">{mhd['suggestion']}</p>
                    <p style="margin:0;font-style:italic;color:var(--text-secondary);">💡 {mhd['wellness_tip']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<h4 style='margin-top:1.5rem;'>Emotion Distribution</h4>", unsafe_allow_html=True)
                emotions_data = [(CLASS_NAMES[i], float(preds[i])) for i in range(len(CLASS_NAMES))]
                emotions_data.sort(key=lambda x: x[1], reverse=True)
                for emo, conf in emotions_data:
                    emo_color = MENTAL_HEALTH_MAPPING.get(emo, {}).get('color', '#6B7280')
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">
                            <span style="font-weight:500;">{MENTAL_HEALTH_MAPPING.get(emo,{}).get('icon','😐')} {emo.capitalize()}</span>
                            <span style="font-weight:700;color:{emo_color};">{conf*100:.1f}%</span>
                        </div>
                        <div class="conf-bar-wrap" style="height:6px;">
                            <div style="height:100%;width:{conf*100:.0f}%;background:{emo_color};border-radius:999px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<h4 style='margin-top:1.5rem;'>Recommended Activities</h4>", unsafe_allow_html=True)
                cols = st.columns(3)
                for i, activity in enumerate(mhd['activities']):
                    with cols[i]:
                        if st.button(f"✅ {activity}", key=f"fac_act_{i}", use_container_width=True):
                            st.success(f"Great! {activity}")

                if st.button("💾 Save Analysis", use_container_width=True, type="primary"):
                    data = {
                        'type': 'facial_analysis', 'dominant_emotion': dominant, 'confidence': max_conf,
                        'all_emotions': {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))},
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.analysis_history.append(data)
                    save_analysis_to_history(st.session_state.username, data)
                    update_streak()
                    check_achievements()
                    st.success("✅ Saved to history!")
                    st.balloons()

# ==================== TEXT ANALYSIS ====================
def show_text_analysis():
    st.markdown("""
    <div class="fade-in">
        <h2 style="font-weight:800;margin-bottom:0.25rem;">📝 Text Emotion Analysis</h2>
        <p style="color:var(--text-secondary);margin-bottom:1.5rem;">Advanced NLP-powered emotion detection, sentiment analysis & crisis assessment</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area("Share your thoughts, journal entry, or any text:", height=220,
            placeholder="Today I felt really anxious about the meeting. My heart was racing and I couldn't focus. But then I took a walk and it helped a lot...")
        c1, c2, c3 = st.columns(3)
        with c1:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
        with c2:
            if st.button("📋 Sample Text", use_container_width=True):
                st.rerun()
        with c3:
            depth = st.selectbox("Depth", ["Standard", "Deep", "Clinical"], label_visibility="collapsed")

    with col2:
        st.markdown("""
        <div class="nw-card-flat">
            <h4 style="margin:0 0 0.75rem 0;font-size:0.9rem;">Text Templates</h4>
        """, unsafe_allow_html=True)
        templates = {
            "Journal Entry": "Today was quite challenging. I felt overwhelmed at work and struggled to find motivation...",
            "Positive Day": "Had an amazing day! Everything went perfectly and I felt so grateful and energized...",
            "Anxious Moment": "I'm really worried about the upcoming exam. My mind keeps racing and I can't sleep...",
            "Grateful": "I'm so thankful for the people in my life. They make everything so much brighter...",
        }
        for tname, ttext in templates.items():
            if st.button(tname, key=f"tmpl_{tname}", use_container_width=True):
                st.session_state['template_text'] = ttext
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if 'template_text' in st.session_state and not text_input:
        text_input = st.session_state.template_text

    if text_input:
        words = len(text_input.split())
        chars = len(text_input)
        st.markdown(f"""
        <div style="display:flex;gap:1rem;margin-bottom:0.5rem;">
            <span class="badge badge-primary">📝 {words} words</span>
            <span class="badge badge-neutral">🔤 {chars} chars</span>
            <span class="badge badge-primary">{'✅ Ready' if words >= 3 else '⚠️ Too short'}</span>
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn and text_input:
        if len(text_input.split()) < 3:
            st.warning("Please enter at least 3 words for meaningful analysis.")
        else:
            with st.spinner("Performing comprehensive text analysis..."):
                result = text_analyzer.analyze_text(text_input)
                result['type'] = 'text_analysis'
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result['context'] = 'General'

            # Crisis alert
            crisis = result['crisis_risk']
            if crisis['risk_level'] != 'none':
                risk_colors = {'CRITICAL': '#7F1D1D', 'HIGH': '#EF4444', 'MODERATE': '#F59E0B', 'LOW': '#FBBF24'}
                rc = risk_colors.get(crisis['risk_level'], '#6B7280')
                st.markdown(f"""
                <div class="crisis-{crisis['risk_level'].lower()}" style="margin-bottom:1rem;">
                    <h3 style="color:{rc};margin:0;">⚠️ {crisis['risk_level']} RISK DETECTED</h3>
                    <p style="margin:0.5rem 0 0 0;">Severity Score: {crisis['severity_score']}/10 · Indicators: {crisis['indicator_count']}</p>
                </div>
                """, unsafe_allow_html=True)
                if crisis['requires_immediate_attention']:
                    with st.expander("📞 Crisis Resources — Click to see"):
                        st.markdown("""
                        **National Suicide Prevention Lifeline:** 988 or 1-800-273-8255
                        **Crisis Text Line:** Text HOME to 741741
                        **SAMHSA National Helpline:** 1-800-662-4357
                        **Emergency:** 911
                        """)

            st.markdown("---")
            st.markdown("<h3 style='font-weight:700;'>Analysis Results</h3>", unsafe_allow_html=True)

            # Main metrics
            sentiment = result['sentiment']
            stats = result['statistics']
            wellness = result['wellness']
            dominant = result['dominant_emotion']
            color = wellness.get('color', '#6B7280')

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="nw-card" style="text-align:center;border-top:4px solid {color};">
                    <div style="font-size:3rem;">{wellness.get('icon','😐')}</div>
                    <div style="font-size:1.2rem;font-weight:800;color:{color};">{dominant.title()}</div>
                    <div style="font-size:0.8rem;color:var(--text-secondary);">{result['dominant_score']*100:.0f}% intensity</div>
                    <div class="conf-bar-wrap" style="margin-top:0.5rem;">
                        <div class="conf-bar-fill" style="width:{result['dominant_score']*100:.0f}%;background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                sent_val = sentiment['compound']
                sent_color = '#22C55E' if sent_val > 0.1 else '#EF4444' if sent_val < -0.1 else '#6B7280'
                st.metric("Sentiment", f"{sent_val:.2f}", delta="Positive" if sent_val > 0 else "Negative" if sent_val < 0 else "Neutral")
            with col3:
                st.metric("Word Count", stats['word_count'])
            with col4:
                st.metric("Lexical Diversity", f"{stats['lexical_diversity']:.2f}")

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                # Sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment['compound'] * 100,
                    title={'text': "Sentiment Score", 'font': {'color': get_colors()['text']}},
                    gauge={
                        'axis': {'range': [-100, 100], 'tickcolor': get_colors()['text_secondary']},
                        'bar': {'color': '#6366F1'},
                        'steps': [
                            {'range': [-100, -30], 'color': '#FEE2E2'},
                            {'range': [-30, 30], 'color': '#F3F4F6'},
                            {'range': [30, 100], 'color': '#DCFCE7'}
                        ],
                    }
                ))
                fig.update_layout(height=200, margin=dict(t=50, b=10, l=10, r=10),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color=get_colors()['text']))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Emotion distribution
                emotions = result['emotions_detected']
                if emotions:
                    emo_df = pd.DataFrame([{'Emotion': e.capitalize(), 'Score': s*100} for e, s in sorted(emotions.items(), key=lambda x: x[1], reverse=True)])
                    fig2 = px.bar(emo_df, x='Score', y='Emotion', orientation='h',
                                  color='Score', color_continuous_scale='Viridis', range_color=[0,100])
                    fig2.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10),
                                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                       font=dict(color=get_colors()['text']), showlegend=False,
                                       coloraxis_showscale=False, yaxis_title='', xaxis_title='Intensity (%)')
                    st.plotly_chart(fig2, use_container_width=True)

            # Wellness recommendation
            st.markdown(f"""
            <div class="wellness-box" style="border-left-color:{color};">
                <h4 style="color:{color};margin:0 0 0.5rem 0;">{wellness.get('status','Neutral')}</h4>
                <p style="margin:0 0 0.5rem 0;">{wellness.get('suggestion','')}</p>
                <p style="margin:0;font-style:italic;color:var(--text-secondary);">💡 {wellness.get('wellness_tip','')}</p>
            </div>
            """, unsafe_allow_html=True)

            if depth in ["Deep", "Clinical"]:
                with st.expander("🔬 Deep Analysis"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Text Statistics**")
                        st.markdown(f"- Words: {stats['word_count']} (Unique: {stats['unique_words']})")
                        st.markdown(f"- Avg word length: {stats['avg_word_length']:.1f} chars")
                        st.markdown(f"- Avg sentence: {stats['avg_sentence_length']:.1f} words")
                    with c2:
                        if depth == "Clinical":
                            st.markdown("**POS Analysis**")
                            st.markdown(f"- Nouns: {stats['noun_count']} | Verbs: {stats['verb_count']}")
                            st.markdown(f"- Adjectives: {stats['adj_count']} | Adverbs: {stats['adv_count']}")
                            st.markdown(f"- Pronouns: {stats['pronoun_count']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save to History", use_container_width=True, type="primary"):
                    st.session_state.analysis_history.append(result)
                    save_analysis_to_history(st.session_state.username, result)
                    update_streak()
                    check_achievements()
                    st.success("✅ Saved!")
            with col2:
                if st.button("📄 Export Report", use_container_width=True):
                    pdf = generate_pdf_report([result], st.session_state.username, "text")
                    st.download_button("📥 Download PDF", pdf, f"neurowell_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf")

# ==================== VOICE ANALYSIS ====================
def show_voice_analysis():
    st.markdown("""
    <div class="fade-in">
        <h2 style="font-weight:800;margin-bottom:0.25rem;">🎤 Voice Emotion Analysis</h2>
        <p style="color:var(--text-secondary);margin-bottom:1.5rem;">ML-based emotion detection with 20+ acoustic features & voice health monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎙️ Record Voice", "📁 Upload Audio"])

    def analyze_audio(path):
        features, y, sr = voice_analyzer.extract_enhanced_features(path)
        if features is None: return None
        dominant, confidence, all_emotions = voice_analyzer.detect_emotion_ml(features)
        health = voice_analyzer.analyze_voice_health_advanced(features)
        emotion_data = VOICE_EMOTION_MAPPING.get(dominant, VOICE_EMOTION_MAPPING['neutral'])
        duration = len(y) / sr if y is not None else 0
        return {
            'type': 'voice_analysis', 'dominant_emotion': dominant, 'confidence': confidence,
            'all_emotions': all_emotions, 'features': features, 'health_indicators': health,
            'emotion_data': emotion_data, 'duration': duration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'voice_quality': {
                'clarity': 'Excellent' if features.get('zcr_mean', 0) < 0.08 else 'Good' if features.get('zcr_mean', 0) < 0.15 else 'Moderate',
                'stability': 'Very Stable' if features.get('pitch_std', 100) < 30 else 'Stable' if features.get('pitch_std', 100) < 60 else 'Variable',
                'energy_level': 'High' if features.get('energy_mean', 0) > 0.15 else 'Moderate' if features.get('energy_mean', 0) > 0.05 else 'Low'
            }
        }

    def display_voice_results(result):
        if not result:
            st.error("Analysis failed. Ensure audio is clear and at least 2 seconds.")
            return
        emotion_data = result['emotion_data']
        dominant = result['dominant_emotion']
        confidence = result['confidence']
        color = emotion_data['color']

        st.markdown("---")
        st.markdown("<h3 style='font-weight:700;'>Voice Analysis Results</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            conf_color = '#22C55E' if confidence > 0.7 else '#F59E0B' if confidence > 0.5 else '#EF4444'
            st.markdown(f"""
            <div class="emotion-hero" style="background:linear-gradient(135deg,{color}20,{color}40);border:2px solid {color}40;">
                <div style="font-size:4rem;">{emotion_data['icon']}</div>
                <div style="font-size:1.6rem;font-weight:900;color:{color};margin:0.5rem 0;">{dominant.upper()}</div>
                <div style="font-size:1.4rem;font-weight:800;color:{conf_color};">{confidence*100:.1f}%</div>
                <div style="font-size:0.8rem;color:var(--text-secondary);">Confidence</div>
                <div class="conf-bar-wrap" style="margin-top:0.75rem;">
                    <div class="conf-bar-fill" style="width:{confidence*100:.0f}%;background:{color};"></div>
                </div>
                <div style="margin-top:0.75rem;">
                    <span class="badge {'badge-positive' if emotion_data['category']=='Positive' else 'badge-negative' if emotion_data['category']=='Negative' else 'badge-neutral'}">{emotion_data['category']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="wellness-box">
                <h4 style="color:var(--primary);margin:0 0 0.5rem 0;">{emotion_data['status']}</h4>
                <p style="margin:0 0 0.5rem 0;"><strong>Clinical:</strong> {emotion_data.get('clinical_term','N/A')}</p>
                <p style="margin:0 0 0.5rem 0;">{emotion_data['suggestion']}</p>
                <p style="margin:0;font-style:italic;color:var(--text-secondary);">💡 {emotion_data['wellness_tip']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Voice metrics
            features = result['features']
            quality = result.get('voice_quality', {})
            st.markdown("<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.75rem;'>", unsafe_allow_html=True)
            for label, value in [
                ("Duration", f"{result['duration']:.1f}s"),
                ("Avg Pitch", f"{features.get('pitch_mean',0):.0f} Hz"),
                ("Energy", features.get('energy_level', quality.get('energy_level','N/A'))),
                ("Clarity", quality.get('clarity','N/A')),
                ("Stability", quality.get('stability','N/A')),
                ("Speech Rate", f"{features.get('speech_rate',0):.0f} spm"),
            ]:
                st.markdown(f"""
                <div style="background:var(--surface2);border-radius:10px;padding:0.6rem;text-align:center;">
                    <div style="font-weight:700;color:var(--primary);font-size:0.95rem;">{value}</div>
                    <div style="font-size:0.7rem;color:var(--text-secondary);text-transform:uppercase;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Emotion distribution
        st.markdown("<h4 style='margin-top:1.5rem;'>Emotion Probabilities</h4>", unsafe_allow_html=True)
        all_emo = result['all_emotions']
        for emo, score in sorted(all_emo.items(), key=lambda x: x[1], reverse=True):
            emo_color = VOICE_EMOTION_MAPPING.get(emo, {}).get('color', '#6B7280')
            emo_icon = VOICE_EMOTION_MAPPING.get(emo, {}).get('icon', '😐')
            st.markdown(f"""
            <div style="margin-bottom:0.4rem;">
                <div style="display:flex;justify-content:space-between;">
                    <span>{emo_icon} {emo.capitalize()}</span>
                    <span style="font-weight:700;color:{emo_color};">{score*100:.1f}%</span>
                </div>
                <div class="conf-bar-wrap" style="height:5px;">
                    <div style="height:100%;width:{score*100:.0f}%;background:{emo_color};border-radius:999px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Health indicators
        if result['health_indicators']:
            st.markdown("<h4 style='margin-top:1.5rem;'>🏥 Voice Health Alerts</h4>", unsafe_allow_html=True)
            for ind in result['health_indicators']:
                sev_color = {'high': '#EF4444', 'moderate': '#F59E0B'}.get(ind['severity'], '#6B7280')
                with st.expander(f"⚠️ {ind['issue']} ({ind['severity'].upper()})"):
                    st.markdown(f"""
                    <div style="padding:0.75rem;background:{sev_color}10;border-left:4px solid {sev_color};border-radius:0 8px 8px 0;">
                        <p><strong>Clinical:</strong> {ind.get('clinical_term','N/A')}</p>
                        <p>{ind['suggestion']}</p>
                        <p><strong>Exercises:</strong> {', '.join(ind.get('exercises',[]))}</p>
                    </div>
                    """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Analysis", use_container_width=True, type="primary", key="save_voice"):
                st.session_state.analysis_history.append(result)
                save_analysis_to_history(st.session_state.username, result)
                update_streak()
                check_achievements()
                st.success("✅ Saved!")
        with col2:
            if st.button("📄 Export PDF", use_container_width=True, key="export_voice"):
                pdf = generate_pdf_report([result], st.session_state.username, "voice")
                st.download_button("📥 Download", pdf, f"neurowell_voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf")

    with tab1:
        st.markdown("""
        <div class="voice-recorder">
            <div style="font-size:3rem;margin-bottom:1rem;">🎙️</div>
            <h3 style="margin:0 0 0.5rem 0;">Record Your Voice</h3>
            <p style="color:var(--text-secondary);margin:0;">Speak naturally for 5–10 seconds. Minimize background noise for best results.</p>
        </div>
        """, unsafe_allow_html=True)
        audio_bytes = st.audio_input("Record a voice message")
        if audio_bytes:
            with st.spinner("🔊 Analyzing voice with enhanced ML algorithms..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_bytes.read())
                    tmp_path = tmp.name
                result = analyze_audio(tmp_path)
                try: os.unlink(tmp_path)
                except: pass
            display_voice_results(result)

    with tab2:
        uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
        if uploaded:
            with st.spinner("🔊 Analyzing uploaded audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                result = analyze_audio(tmp_path)
                try: os.unlink(tmp_path)
                except: pass
            display_voice_results(result)

    with st.expander("💡 Tips for Accurate Analysis"):
        st.markdown("""
        - **Speak clearly** at a natural pace (5–10 seconds minimum)
        - **Minimize background noise** — quiet environments give best results
        - **Use a good microphone** — phone mic or headset works great
        - **Speak naturally** — don't force or exaggerate emotions
        - **Supported formats:** WAV (recommended), MP3, M4A, OGG, FLAC
        """)

# ==================== WELLNESS TIPS ====================
def show_wellness_tips():
    st.markdown("""
    <div class="fade-in">
        <h2 style="font-weight:800;margin-bottom:0.25rem;">💚 Wellness Tips</h2>
        <p style="color:var(--text-secondary);margin-bottom:1.5rem;">Personalized mental health insights for every emotional state</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎭 Facial Emotions", "📝 Text Emotions", "🎤 Voice Emotions"])

    def render_tips(mapping, filter_options):
        selected = st.selectbox("Filter by emotion", ["All"] + [e.capitalize() for e in filter_options])
        to_show = filter_options if selected == "All" else [selected.lower()]
        for emotion in to_show:
            data = mapping.get(emotion, {})
            color = data.get('color', '#6B7280')
            icon = data.get('icon', '😐')
            with st.expander(f"{icon}  {emotion.capitalize()} — {data.get('status', '')}"):
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f"""
                    <div class="wellness-box" style="border-left-color:{color};">
                        <p><strong>💡 Insight:</strong> {data.get('wellness_tip','')}</p>
                        <p><strong>📋 Suggestion:</strong> {data.get('suggestion','')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("**Recommended Activities:**")
                    for act in data.get('activities', []):
                        st.markdown(f"• {act}")
                with col2:
                    cat = data.get('category', 'Neutral')
                    badge_cls = 'badge-positive' if cat == 'Positive' else 'badge-negative' if cat == 'Negative' else 'badge-neutral'
                    st.markdown(f"""
                    <div style="background:{color}10;border:1px solid {color}30;border-radius:16px;padding:1.25rem;text-align:center;">
                        <div style="font-size:3rem;">{icon}</div>
                        <div style="font-weight:700;color:{color};margin:0.5rem 0;">{emotion.capitalize()}</div>
                        <span class="badge {badge_cls}">{cat}</span>
                        {'<div style="margin-top:0.75rem;font-size:0.8rem;color:var(--text-secondary);">Clinical: ' + data.get('clinical_term','') + '</div>' if 'clinical_term' in data else ''}
                    </div>
                    """, unsafe_allow_html=True)

    with tab1:
        render_tips(MENTAL_HEALTH_MAPPING, CLASS_NAMES)
    with tab2:
        render_tips(TEXT_EMOTION_MAPPING, list(TEXT_EMOTION_MAPPING.keys()))
    with tab3:
        render_tips(VOICE_EMOTION_MAPPING, list(VOICE_EMOTION_MAPPING.keys()))

# ==================== HISTORY ====================
def show_history():
    st.markdown("""
    <div class="fade-in">
        <h2 style="font-weight:800;margin-bottom:0.25rem;">📋 Analysis History</h2>
        <p style="color:var(--text-secondary);margin-bottom:1.5rem;">All your past emotion analyses in one place</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_history:
        st.markdown("""
        <div class="nw-card" style="text-align:center;padding:3rem;">
            <div style="font-size:4rem;">📭</div>
            <h3>No History Yet</h3>
            <p style="color:var(--text-secondary);">Complete some analyses to see them here!</p>
        </div>
        """, unsafe_allow_html=True)
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox("Type", ["All", "Facial", "Text", "Voice"])
    with col2:
        sort_order = st.selectbox("Sort", ["Newest First", "Oldest First"])
    with col3:
        search = st.text_input("Search emotion", placeholder="e.g. happy")

    filtered = []
    for e in st.session_state.analysis_history:
        t = e.get('type', 'facial_analysis')
        if type_filter != "All":
            if type_filter == "Facial" and t != 'facial_analysis': continue
            if type_filter == "Text" and t != 'text_analysis': continue
            if type_filter == "Voice" and t != 'voice_analysis': continue
        if search and search.lower() not in e.get('dominant_emotion', '').lower(): continue
        filtered.append(e)

    if sort_order == "Newest First":
        filtered = list(reversed(filtered))

    st.markdown(f"""
    <div class="nw-card-flat" style="margin-bottom:1rem;">
        <span class="badge badge-primary">📊 {len(filtered)} results</span>
        <span class="badge badge-neutral" style="margin-left:0.5rem;">of {len(st.session_state.analysis_history)} total</span>
    </div>
    """, unsafe_allow_html=True)

    for i, entry in enumerate(filtered):
        t = entry.get('type', 'facial_analysis')
        emotion = entry.get('dominant_emotion', 'neutral')
        ts = entry.get('timestamp', 'N/A')
        conf = entry.get('confidence', 0)
        color = MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#6B7280')
        icon = MENTAL_HEALTH_MAPPING.get(emotion, {}).get('icon', '😐')
        type_icon = '🎭' if t == 'facial_analysis' else '📝' if t == 'text_analysis' else '🎤'
        type_label = t.replace('_', ' ').title()
        cat = MENTAL_HEALTH_MAPPING.get(emotion, {}).get('category', 'Neutral')
        badge_cls = 'badge-positive' if cat == 'Positive' else 'badge-negative' if cat == 'Negative' else 'badge-neutral'

        with st.expander(f"{type_icon} {type_label} — {icon} {emotion.capitalize()} — {ts[:16] if len(ts)>16 else ts}"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.75rem;">
                    <div style="font-size:2.5rem;">{icon}</div>
                    <div>
                        <div style="font-size:1.2rem;font-weight:700;color:{color};">{emotion.upper()}</div>
                        <div style="display:flex;gap:0.5rem;margin-top:0.25rem;">
                            <span class="badge {badge_cls}">{cat}</span>
                            <span class="badge badge-primary">{conf*100:.0f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if t == 'text_analysis':
                    st.markdown(f"**Preview:** {entry.get('text_preview','N/A')[:80]}...")
                    sent = entry.get('sentiment', {})
                    if isinstance(sent, dict):
                        score = sent.get('compound', 0)
                        st.markdown(f"**Sentiment:** {score:.2f}")
                elif t == 'voice_analysis':
                    features = entry.get('features', {})
                    st.markdown(f"**Pitch:** {features.get('pitch_mean',0):.0f} Hz | **Duration:** {entry.get('duration',0):.1f}s")
                else:
                    st.markdown(f"**Type:** Facial Expression Analysis")
                    all_emo = entry.get('all_emotions', {})
                    if all_emo:
                        top2 = sorted(all_emo.items(), key=lambda x: x[1], reverse=True)[:2]
                        st.markdown(f"**Top emotions:** {', '.join([f'{e}({s*100:.0f}%)' for e,s in top2])}")
            with col3:
                if st.button("🗑️ Delete", key=f"del_{i}_{ts}", use_container_width=True):
                    st.session_state.analysis_history.remove(entry)
                    try:
                        with open(USER_DB_FILE, 'r') as f:
                            users = json.load(f)
                        if st.session_state.username in users:
                            users[st.session_state.username]['history'] = [
                                h for h in users[st.session_state.username].get('history', [])
                                if h.get('data', {}).get('timestamp') != entry.get('timestamp')
                            ]
                            with open(USER_DB_FILE, 'w') as f:
                                json.dump(users, f, indent=4)
                    except: pass
                    st.rerun()

# ==================== ANALYTICS ====================
def show_analytics():
    st.markdown("""
    <div class="fade-in">
        <h2 style="font-weight:800;margin-bottom:0.25rem;">📈 Analytics Dashboard</h2>
        <p style="color:var(--text-secondary);margin-bottom:1.5rem;">Visualize your emotional patterns and wellness trends</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_history:
        st.markdown("""
        <div class="nw-card" style="text-align:center;padding:3rem;">
            <div style="font-size:4rem;">📊</div>
            <h3>No Data Available</h3>
            <p style="color:var(--text-secondary);">Complete some analyses to see your dashboard!</p>
        </div>
        """, unsafe_allow_html=True)
        return

    history = st.session_state.analysis_history
    facial = [e for e in history if e.get('type','facial_analysis') == 'facial_analysis']
    text_d = [e for e in history if e.get('type') == 'text_analysis']
    voice_d = [e for e in history if e.get('type') == 'voice_analysis']

    # Summary metrics
    wellness_score, trend = get_wellness_score()
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("Total", len(history), "var(--primary)"),
        ("Facial", len(facial), "#6366F1"),
        ("Text", len(text_d), "#10B981"),
        ("Voice", len(voice_d), "#F59E0B"),
        ("Wellness", f"{wellness_score}/100", "#22C55E"),
    ]
    for col, (label, val, color) in zip([col1,col2,col3,col4,col5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-pill" style="border:1px solid {color}30;">
                <span class="value" style="color:{color};">{val}</span>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        # Emotion distribution pie
        all_emotions = [e.get('dominant_emotion', 'neutral') for e in history]
        if all_emotions:
            counts = Counter(all_emotions)
            colors_list = [MENTAL_HEALTH_MAPPING.get(e, {}).get('color', '#6B7280') for e in counts.keys()]
            fig = go.Figure(go.Pie(
                labels=[e.capitalize() for e in counts.keys()],
                values=list(counts.values()),
                marker=dict(colors=colors_list),
                hole=0.5,
                textinfo='label+percent'
            ))
            fig.update_layout(
                title="Overall Emotion Distribution",
                height=350, margin=dict(t=50,b=10,l=10,r=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=get_colors()['text']),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Analysis type breakdown
        type_counts = {'Facial': len(facial), 'Text': len(text_d), 'Voice': len(voice_d)}
        type_counts = {k: v for k, v in type_counts.items() if v > 0}
        if type_counts:
            fig2 = go.Figure(go.Bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                marker=dict(color=['#6366F1', '#10B981', '#F59E0B'][:len(type_counts)]),
                text=list(type_counts.values()),
                textposition='outside'
            ))
            fig2.update_layout(
                title="Analyses by Type",
                height=350, margin=dict(t=50,b=30,l=30,r=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=get_colors()['text']),
                xaxis=dict(gridcolor=get_colors()['border']),
                yaxis=dict(gridcolor=get_colors()['border'])
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Timeline if enough data
    entries_with_ts = [e for e in history if e.get('timestamp')]
    if len(entries_with_ts) >= 2:
        st.markdown("<h3 style='font-weight:700;margin-top:1rem;'>Emotion Timeline</h3>", unsafe_allow_html=True)
        df_timeline = pd.DataFrame([{
            'datetime': pd.to_datetime(e['timestamp'], errors='coerce'),
            'emotion': e.get('dominant_emotion', 'neutral'),
            'confidence': e.get('confidence', 0.5),
            'type': e.get('type', 'facial_analysis')
        } for e in entries_with_ts]).dropna(subset=['datetime'])

        if not df_timeline.empty:
            df_timeline = df_timeline.sort_values('datetime')
            fig3 = go.Figure()
            for emotion in df_timeline['emotion'].unique():
                edf = df_timeline[df_timeline['emotion'] == emotion]
                color = MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#6B7280')
                fig3.add_trace(go.Scatter(
                    x=edf['datetime'], y=edf['confidence'],
                    mode='markers+lines', name=emotion.capitalize(),
                    marker=dict(size=10, color=color),
                    line=dict(color=color, width=2, dash='dot'),
                    hovertemplate=f"<b>{emotion.capitalize()}</b><br>%{{x}}<br>Confidence: %{{y:.0%}}<extra></extra>"
                ))
            fig3.update_layout(
                height=380, margin=dict(t=30,b=30,l=30,r=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=get_colors()['text']),
                xaxis=dict(gridcolor=get_colors()['border']),
                yaxis=dict(gridcolor=get_colors()['border'], tickformat='.0%', range=[0,1]),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig3, use_container_width=True)

    # Text sentiment trend
    if text_d:
        st.markdown("<h3 style='font-weight:700;'>Text Sentiment Trend</h3>", unsafe_allow_html=True)
        sent_data = []
        for e in text_d:
            if e.get('timestamp') and isinstance(e.get('sentiment'), dict):
                sent_data.append({
                    'datetime': pd.to_datetime(e['timestamp'], errors='coerce'),
                    'sentiment': e['sentiment'].get('compound', 0),
                    'emotion': e.get('dominant_emotion', 'neutral')
                })
        if sent_data:
            df_sent = pd.DataFrame(sent_data).dropna(subset=['datetime']).sort_values('datetime')
            fig_sent = go.Figure()
            fig_sent.add_hline(y=0, line_dash="dot", line_color=get_colors()['text_secondary'])
            fig_sent.add_trace(go.Scatter(
                x=df_sent['datetime'], y=df_sent['sentiment'],
                mode='lines+markers',
                line=dict(color='#6366F1', width=3),
                marker=dict(size=8, color=['#22C55E' if s > 0 else '#EF4444' for s in df_sent['sentiment']]),
                fill='tonexty' if len(df_sent) > 1 else None,
                fillcolor='rgba(99,102,241,0.1)'
            ))
            fig_sent.update_layout(
                height=300, margin=dict(t=20,b=30,l=30,r=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=get_colors()['text']),
                xaxis=dict(gridcolor=get_colors()['border']),
                yaxis=dict(gridcolor=get_colors()['border'], range=[-1,1], title='Sentiment')
            )
            st.plotly_chart(fig_sent, use_container_width=True)

    # Export section
    st.markdown("---")
    st.markdown("<h3 style='font-weight:700;'>Export Reports</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
            with st.spinner("Generating report..."):
                pdf = generate_pdf_report(history, st.session_state.username)
                st.download_button(
                    "📥 Download PDF",
                    pdf,
                    f"neurowell_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "application/pdf",
                    key="dl_pdf"
                )
            st.success("✅ Report ready!")
    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            rows = []
            for e in history:
                rows.append({'Date': e.get('timestamp','')[:10], 'Type': e.get('type',''), 'Emotion': e.get('dominant_emotion',''), 'Confidence': f"{e.get('confidence',0)*100:.1f}%"})
            df_export = pd.DataFrame(rows)
            csv = df_export.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"neurowell_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    with col3:
        if st.button("📦 Export JSON", use_container_width=True):
            json_str = json.dumps(history, indent=2, default=str)
            st.download_button("📥 Download JSON", json_str, f"neurowell_{datetime.now().strftime('%Y%m%d')}.json", "application/json")

# ==================== PROFILE ====================
def show_profile():
    C = get_colors()
    st.markdown("""
    <div class="fade-in">
        <h2 style="font-weight:800;margin-bottom:0.25rem;">👤 Your Profile</h2>
        <p style="color:var(--text-secondary);margin-bottom:1.5rem;">Manage your account, settings, and data</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        user_data = users.get(st.session_state.username, {})
    except:
        user_data = {}

    tab1, tab2, tab3, tab4 = st.tabs(["👤 Profile", "⚙️ Settings", "🏆 Achievements", "🗄️ Data"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            profile_html = get_profile_img_html(st.session_state.username, 120)
            st.markdown(f"""
            <div class="nw-card" style="text-align:center;">
                {profile_html}
                <h3 style="margin:1rem 0 0.25rem 0;">{st.session_state.username}</h3>
                <p style="color:var(--text-secondary);margin:0;">{user_data.get('email','')}</p>
                <div style="margin-top:0.75rem;">
                    <span class="streak-badge">🔥 {st.session_state.get('streak',0)} day streak</span>
                </div>
                <div style="margin-top:0.75rem;font-size:0.8rem;color:var(--text-secondary);">
                    Member since {user_data.get('created_at','N/A')[:10]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<h4>Upload Photo</h4>", unsafe_allow_html=True)
            uploaded_img = st.file_uploader("Choose image", type=["jpg","jpeg","png"], label_visibility="collapsed")
            if uploaded_img:
                st.image(uploaded_img, width=150)
                if st.button("💾 Save Photo", use_container_width=True):
                    ok, res = save_profile_image(st.session_state.username, uploaded_img.getvalue())
                    if ok:
                        st.session_state.profile_image = res
                        st.success("✅ Photo updated!")
                        st.rerun()
                    else:
                        st.error(f"Failed: {res}")

        with col2:
            wellness_score, trend = get_wellness_score()
            history = st.session_state.analysis_history
            col_a, col_b = st.columns(2)
            for col, (label, val, color) in zip(
                [col_a, col_b, col_a, col_b],
                [
                    ("Total Analyses", len(history), "var(--primary)"),
                    ("Wellness Score", f"{wellness_score}/100", '#22C55E'),
                    ("Achievements", len(st.session_state.get('achievements',[])), "var(--accent)"),
                    ("Trend", trend.replace('_',' ').title(), "var(--secondary)"),
                ]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-pill" style="margin-bottom:0.75rem;">
                        <span class="value" style="color:{color};">{val}</span>
                        <div class="label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h4>Appearance</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            dm = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
            if dm != st.session_state.dark_mode:
                st.session_state.dark_mode = dm
                st.rerun()
        with col2:
            fs = st.select_slider("Font Size", ["small", "medium", "large"], value=st.session_state.font_size)
            if fs != st.session_state.font_size:
                st.session_state.font_size = fs
                st.rerun()

        st.markdown("<h4>Notifications</h4>", unsafe_allow_html=True)
        notifs = st.toggle("Email Notifications", value=user_data.get('settings', {}).get('notifications', True))
        save_hist = st.toggle("Auto-save History", value=user_data.get('settings', {}).get('save_history', True))

        if st.button("💾 Save Settings", type="primary"):
            settings = {'notifications': notifs, 'save_history': save_hist, 'dark_mode': st.session_state.dark_mode, 'font_size': st.session_state.font_size}
            if update_user_settings(st.session_state.username, settings):
                st.success("✅ Settings saved!")
            else:
                st.error("Failed to save settings")

    with tab3:
        earned = set(st.session_state.get('achievements', []))
        cols = st.columns(3)
        for i, (ach_id, ach) in enumerate(ACHIEVEMENTS.items()):
            with cols[i % 3]:
                unlocked = ach_id in earned
                opacity = "1" if unlocked else "0.35"
                st.markdown(f"""
                <div class="achievement" style="opacity:{opacity};margin-bottom:0.75rem;">
                    <div class="ach-icon">{ach['icon']}</div>
                    <div class="ach-name">{ach['name']}</div>
                    <div class="ach-desc">{ach['desc']}</div>
                    {'<div style="color:var(--success);font-size:0.75rem;font-weight:700;margin-top:0.3rem;">✓ UNLOCKED</div>' if unlocked else '<div style="color:var(--text-secondary);font-size:0.7rem;margin-top:0.3rem;">🔒 Locked</div>'}
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export My Data", use_container_width=True):
                export = {'username': st.session_state.username, 'history': st.session_state.analysis_history, 'achievements': st.session_state.get('achievements', [])}
                st.download_button("📥 Download JSON", json.dumps(export, indent=2, default=str), f"neurowell_{st.session_state.username}.json", "application/json")
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                st.session_state.analysis_history = []
                try:
                    with open(USER_DB_FILE, 'r') as f:
                        users = json.load(f)
                    users[st.session_state.username]['history'] = []
                    with open(USER_DB_FILE, 'w') as f:
                        json.dump(users, f, indent=4)
                except: pass
                st.success("History cleared!")
                st.rerun()

        st.markdown("---")
        st.markdown("<h4 style='color:var(--danger);'>⚠️ Danger Zone</h4>", unsafe_allow_html=True)
        with st.expander("Delete Account"):
            st.warning("This action cannot be undone. All your data will be permanently deleted.")
            confirm = st.text_input("Type 'DELETE' to confirm")
            if st.button("🗑️ Delete Account", type="primary"):
                if confirm == "DELETE":
                    try:
                        with open(USER_DB_FILE, 'r') as f:
                            users = json.load(f)
                        del users[st.session_state.username]
                        with open(USER_DB_FILE, 'w') as f:
                            json.dump(users, f, indent=4)
                    except: pass
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    init_session()
                    st.rerun()
                else:
                    st.error("Please type 'DELETE' to confirm")

# ==================== MAIN APP ====================
def show_main_app():
    show_sidebar()

    page = st.session_state.current_page
    if page == "Home": show_home()
    elif page == "Facial Analysis": show_facial_analysis()
    elif page == "Text Analysis": show_text_analysis()
    elif page == "Voice Analysis": show_voice_analysis()
    elif page == "Wellness Tips": show_wellness_tips()
    elif page == "History": show_history()
    elif page == "Analytics": show_analytics()
    elif page == "Profile": show_profile()

# ==================== ENTRY POINT ====================
def main():
    if not st.session_state.logged_in:
        show_login_signup()
    else:
        show_main_app()

if __name__ == "__main__":
    main()