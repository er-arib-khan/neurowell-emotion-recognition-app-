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

# ==================== PROFESSIONAL COLOR PALETTE ====================
COLORS = {
    'primary': '#6366F1',  # Indigo
    'primary_dark': '#4F46E5',
    'primary_light': '#818CF8',
    'secondary': '#10B981',  # Emerald
    'accent': '#F59E0B',  # Amber
    'danger': '#EF4444',  # Red
    'warning': '#F97316',  # Orange
    'success': '#22C55E',  # Green
    'info': '#3B82F6',  # Blue
    'dark': '#1F2937',  # Gray 800
    'darker': '#111827',  # Gray 900
    'light': '#F9FAFB',  # Gray 50
    'lighter': '#FFFFFF',  # White
    'gray': '#6B7280',  # Gray 500
    'gray_light': '#E5E7EB',  # Gray 200
    'gray_dark': '#4B5563',  # Gray 600
    'hospital_blue': '#005EB8',  # NHS Blue
    'hospital_grey': '#425563',  # Slate Grey
    'hospital_warm': '#FFB81C',  # Warm Yellow
}

EMOTION_COLORS = {
    'angry': '#EF4444',  # Red
    'disgust': '#10B981',  # Emerald
    'fear': '#8B5CF6',  # Violet
    'happy': '#F59E0B',  # Amber
    'neutral': '#6B7280',  # Gray
    'sad': '#3B82F6',  # Blue
    'surprise': '#EC4899'  # Pink
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
    custom_logo_path = "logo.png"  # Your logo file name
    
    # Check if custom logo exists
    if os.path.exists(custom_logo_path):
        logo_base64 = get_base64_of_image(custom_logo_path)
        if logo_base64:
            # Size mapping
            sizes = {
                "small": "60px",
                "medium": "120px",
                "large": "180px",
                "xlarge": "300px",
                "sidebar": "160px",
                "header": "150px"
                
            }
            return f'<img src="data:image/png;base64,{logo_base64}" style="width: {sizes.get(size, "80px")}; height: auto; object-fit: contain;">'
    
    # Fallback to default icon if custom logo not found
    icon_urls = {
        "small": "https://img.icons8.com/fluency/48/000000/mental-state.png",
        "medium": "https://img.icons8.com/fluency/96/000000/mental-state.png",
        "large": "https://img.icons8.com/fluency/240/000000/mental-state.png",
        "xlarge": "https://img.icons8.com/fluency/480/000000/mental-state.png",
        "sidebar": "https://img.icons8.com/fluency/96/000000/mental-state.png",
        "header": "https://img.icons8.com/fluency/240/000000/mental-state.png",
        "login": "https://img.icons8.com/fluency/240/000000/mental-state.png"

    }

    # Also increase fallback sizes
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

# Create profile images directory if it doesn't exist
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
        # Delete old profile image if exists
        old_pattern = f"{PROFILE_IMAGES_DIR}/{username}_*"
        for old_file in Path(PROFILE_IMAGES_DIR).glob(f"{username}_*"):
            old_file.unlink()
        
        # Save new profile image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"{PROFILE_IMAGES_DIR}/{username}_{timestamp}.png"
        
        # Save image
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        return True, image_path
    except Exception as e:
        return False, str(e)

def get_profile_image(username):
    """Get profile image path for user"""
    try:
        # Look for any profile image with this username
        images = list(Path(PROFILE_IMAGES_DIR).glob(f"{username}_*"))
        if images:
            return str(images[-1])  # Return the most recent
    except:
        pass
    return None

def create_user(username, password, email):
    """Create a new user account"""
    try:
        # Load existing users
        if os.path.exists(USER_DB_FILE):
            with open(USER_DB_FILE, 'r') as f:
                users = json.load(f)
        else:
            users = {}
        
        # Check if username exists
        if username in users:
            return False, "Username already exists!"
        
        # Create new user
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
        
        # Save back to file
        with open(USER_DB_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        
        return True, "Account created successfully!"
    
    except Exception as e:
        return False, f"Error creating account: {str(e)}"

def verify_user(username, password):
    """Verify user credentials"""
    try:
        # Check if file exists
        if not os.path.exists(USER_DB_FILE):
            return False, "No users found. Please sign up first."
        
        # Load users
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        # Check if username exists
        if username not in users:
            return False, "Invalid username or password"
        
        # Verify password
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
            
            # Ensure analysis_data has timestamp
            if 'timestamp' not in analysis_data:
                analysis_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Store in the correct format (with 'data' key)
            users[username]['history'].append({
                'timestamp': analysis_data['timestamp'],
                'data': analysis_data  # Store under 'data' key
            })
            
            # Keep only last 50 analyses
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
            # Ensure we're returning a list
            if isinstance(history, list):
                return history
            else:
                print(f"History is not a list: {type(history)}")
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
                            # Already in correct format
                            new_history.append(entry)
                        elif 'dominant_emotion' in entry:
                            # Old format - convert to new format
                            new_history.append({
                                'timestamp': entry.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                                'data': entry
                            })
                            modified = True
                        else:
                            # Unknown format - skip
                            print(f"Skipping unknown entry format for {username}")
                            modified = True
                    else:
                        # Not a dict - skip
                        print(f"Skipping non-dict entry for {username}")
                        modified = True
                
                if modified:
                    users[username]['history'] = new_history
        
        if modified:
            with open(USER_DB_FILE, 'w') as f:
                json.dump(users, f, indent=4)
            print("History format migration completed")
    except Exception as e:
        print(f"Error during migration: {e}")

# ==================== LOGO CONFIGURATION ====================
LOGO_URLS = {
    "main": "https://iconscout.com/icon/mental-health-awareness-9949594_8045987",      
    "sidebar": "https://iconscout.com/icon/emotion-recognition-11660740_9484583",   
    "header": "https://iconscout.com/icon/mental-health-awareness-9949594_8045987",    
    "large": "https://iconscout.com/icon/mental-health-awareness-9949594_8045987",     
    "favicon": "https://iconscout.com/icon/mental-health-awareness-9949594_8045987"    
}

# Add custom CSS for logo animations
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
        st.error(f"Model file '{model_path}' not found!")
        return None

model = load_emotion_model()

# ==================== CHART GENERATION FUNCTIONS ====================
def create_emotion_timeline(df):
    """Create timeline chart of emotions over time"""
    if df.empty:
        return None
    
    try:
        # Create a copy and ensure datetime
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
        df_copy = df_copy.dropna(subset=['datetime'])
        df_copy = df_copy.sort_values('datetime')
        
        if df_copy.empty:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each emotion
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

def create_emotion_heatmap(df):
    """Create heatmap of emotions by hour and day"""
    if df.empty:
        return None
    
    try:
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'])
        df_copy['hour'] = df_copy['datetime'].dt.hour
        df_copy['day'] = df_copy['datetime'].dt.day_name()
        
        # Create pivot table
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pd.crosstab(
            [df_copy['day'], df_copy['hour']], 
            df_copy['dominant_emotion'],
            normalize='index'
        ).fillna(0)
        
        # Create heatmap for each emotion
        figs = []
        for emotion in CLASS_NAMES:
            if emotion in pivot.columns:
                # Reshape data for heatmap
                heatmap_data = []
                for day in days_order:
                    day_data = []
                    for hour in range(24):
                        try:
                            val = pivot.loc[(day, hour), emotion] * 100
                        except:
                            val = 0
                        day_data.append(val)
                    heatmap_data.append(day_data)
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=[f"{h:02d}:00" for h in range(24)],
                    y=days_order,
                    colorscale='Viridis',
                    colorbar=dict(title="%"),
                    zmin=0,
                    zmax=100
                ))
                
                fig.update_layout(
                    title=f"{emotion.capitalize()} - Time Pattern",
                    xaxis_title="Hour of Day",
                    yaxis_title="Day of Week",
                    height=300
                )
                figs.append(fig)
        
        return figs
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return None

def create_emotion_sunburst(df):
    """Create sunburst chart of emotion distribution"""
    if df.empty:
        return None
    
    try:
        # Prepare data for sunburst
        emotion_counts = df['dominant_emotion'].value_counts()
        
        fig = go.Figure(go.Sunburst(
            labels=[e.capitalize() for e in emotion_counts.index],
            parents=['' for _ in emotion_counts.index],
            values=emotion_counts.values,
            marker=dict(
                colors=[MENTAL_HEALTH_MAPPING.get(e, {}).get('color', '#808080') for e in emotion_counts.index]
            ),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentRoot:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Emotion Distribution Sunburst",
            height=400
        )
        
        return fig
    except Exception as e:
        print(f"Error creating sunburst: {e}")
        return None

def create_emotion_radar(df):
    """Create radar chart of emotion averages"""
    if df.empty:
        return None
    
    try:
        # Calculate average confidence for each emotion
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

def create_emotion_treemap(df):
    """Create treemap of emotions with confidence levels"""
    if df.empty:
        return None
    
    try:
        # Prepare data for treemap
        emotion_data = []
        for emotion in CLASS_NAMES:
            emotion_df = df[df['dominant_emotion'] == emotion]
            if not emotion_df.empty:
                # Split into confidence levels
                high_conf = len(emotion_df[emotion_df['confidence'] >= 0.8])
                med_conf = len(emotion_df[(emotion_df['confidence'] >= 0.5) & (emotion_df['confidence'] < 0.8)])
                low_conf = len(emotion_df[emotion_df['confidence'] < 0.5])
                
                if high_conf > 0:
                    emotion_data.append({
                        'emotion': emotion.capitalize(),
                        'confidence_level': 'High (80-100%)',
                        'count': high_conf,
                        'color': MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#808080')
                    })
                if med_conf > 0:
                    emotion_data.append({
                        'emotion': emotion.capitalize(),
                        'confidence_level': 'Medium (50-79%)',
                        'count': med_conf,
                        'color': MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#808080') + '80'
                    })
                if low_conf > 0:
                    emotion_data.append({
                        'emotion': emotion.capitalize(),
                        'confidence_level': 'Low (0-49%)',
                        'count': low_conf,
                        'color': MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#808080') + '40'
                    })
        
        if not emotion_data:
            return None
        
        df_treemap = pd.DataFrame(emotion_data)
        
        fig = px.treemap(
            df_treemap,
            path=['emotion', 'confidence_level'],
            values='count',
            color='count',
            color_continuous_scale='Viridis',
            title="Emotion Confidence Levels"
        )
        
        fig.update_layout(height=400)
        
        return fig
    except Exception as e:
        print(f"Error creating treemap: {e}")
        return None

def create_emotion_flow(df):
    """Create flow diagram of emotion transitions"""
    if len(df) < 2:
        return None
    
    try:
        # Get emotion transitions
        emotions = df['dominant_emotion'].tolist()
        transitions = []
        
        for i in range(len(emotions) - 1):
            transitions.append((emotions[i], emotions[i + 1]))
        
        # Count transitions
        transition_counts = Counter(transitions)
        
        # Prepare data for sankey diagram
        nodes = list(set([t[0] for t in transition_counts.keys()] + [t[1] for t in transition_counts.keys()]))
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Create sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[n.capitalize() for n in nodes],
                color=[MENTAL_HEALTH_MAPPING.get(n, {}).get('color', '#808080') for n in nodes]
            ),
            link=dict(
                source=[node_indices[t[0]] for t in transition_counts.keys()],
                target=[node_indices[t[1]] for t in transition_counts.keys()],
                value=list(transition_counts.values()),
                color=['rgba(74, 144, 226, 0.2)' for _ in transition_counts]
            )
        )])
        
        fig.update_layout(
            title="Emotion Transition Flow",
            height=500
        )
        
        return fig
    except Exception as e:
        print(f"Error creating flow: {e}")
        return None

def create_weekly_pattern(df):
    """Create weekly pattern chart"""
    if df.empty:
        return None
    
    try:
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'])
        df_copy['day'] = df_copy['datetime'].dt.day_name()
        df_copy['week'] = df_copy['datetime'].dt.isocalendar().week
        
        # Group by day and emotion
        day_emotion = pd.crosstab(df_copy['day'], df_copy['dominant_emotion'])
        
        # Reorder days
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

# ==================== AUTHENTICATION UI ====================
def show_login_signup():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Animated logo for login page - using get_logo_html
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
                                # Load profile image if exists
                                profile_path = get_profile_image(username)
                                if profile_path:
                                    st.session_state.profile_image = profile_path
                                # Load history with proper error handling
                                db_history = load_user_history(username)
                                if db_history:
                                    # Handle different history formats
                                    st.session_state.analysis_history = []
                                    for entry in db_history:
                                        if isinstance(entry, dict):
                                            if 'data' in entry:
                                                # Format: {'timestamp': ..., 'data': {...}}
                                                st.session_state.analysis_history.append(entry['data'])
                                            elif 'dominant_emotion' in entry:
                                                # Format: direct emotion data
                                                st.session_state.analysis_history.append(entry)
                                            else:
                                                # Unknown format, try to salvage what we can
                                                print(f"Unknown history entry format: {entry}")
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

# ==================== MAIN APP ====================
def show_main_app():
    # Sidebar navigation with enhanced logo and profile
    with st.sidebar:
        # Get profile image
        profile_image_path = st.session_state.profile_image or get_profile_image(st.session_state.username)
        
        if profile_image_path and os.path.exists(profile_image_path):
            # Convert image to base64 for display
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
            options=["Home", "Emotion Analysis", "Mental Health Tips", "History", "Analytics", "Profile"],
            icons=["house", "camera", "heart", "clock-history", "graph-up", "person"],
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
    
    # Main content area
    if selected == "Home":
        show_home_page()
    elif selected == "Emotion Analysis":
        show_emotion_analysis()
    elif selected == "Mental Health Tips":
        show_mental_health_tips()
    elif selected == "History":
        show_history()
    elif selected == "Analytics":
        show_analytics()
    elif selected == "Profile":
        show_profile()

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
        Your personal mental health companion that uses advanced facial emotion recognition 
        to help you understand and manage your emotional well-being.
        
        ### 🌟 Key Features:
        - **Real-time Emotion Analysis**: Understand your current emotional state
        - **Mental Health Insights**: Get personalized suggestions based on your emotions
        - **Track Your Journey**: Monitor your emotional patterns over time
        - **Advanced Analytics**: Visualize your emotional trends with multiple chart types
        - **Wellness Tips**: Access curated mental health resources
        
        ### 📊 Supported Emotions:
        """)
        
        # Display emotions in a grid
        cols = st.columns(4)
        emotions_display = [
            ("😠 Angry", "#FF4B4B"),
            ("🤢 Disgust", "#9ACD32"),
            ("😨 Fear", "#800080"),
            ("😊 Happy", "#4CAF50"),
            ("😐 Neutral", "#808080"),
            ("😔 Sad", "#4169E1"),
            ("😲 Surprise", "#FFA500")
        ]
        
        for i, (emotion, color) in enumerate(emotions_display):
            with cols[i % 4]:
                st.markdown(f"<span style='color:{color}; font-size:1.2rem;'>{emotion}</span>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; animation: float 3s ease-in-out infinite;">
            {get_logo_html('xlarge')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌱 Today's Tip")
        st.info("Your emotions are valid messengers, not problems to solve. Listen to what they're telling you.")
        
        # Quick stats
        if st.session_state.analysis_history:
            st.markdown("### 📊 Your Stats")
            st.metric("Total Analyses", len(st.session_state.analysis_history))
            
            # Get most common emotion
            df = pd.DataFrame(st.session_state.analysis_history)
            most_common = df['dominant_emotion'].mode()[0] if not df.empty else "N/A"
            st.metric("Most Common", most_common.capitalize())

def show_emotion_analysis():
    st.markdown("## 🎭 Emotion Analysis")
    st.markdown("Analyze your facial expression to get mental health insights")
    
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
    
    if source and model:
        is_upload_mode = (option == "📁 Upload Image")
        
        with st.spinner("Analyzing..."):
            cropped_face, predictions, error = process_and_predict(source, is_upload=is_upload_mode)
        
        if error:
            st.error(error)
        else:
            # Display results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(cropped_face, caption="Analyzed Face", width=200)
                
                # Confidence meter
                max_conf = np.max(predictions)
                st.markdown(f"### Confidence: {max_conf*100:.1f}%")
                st.progress(min(int(max_conf*100), 100))
            
            with col2:
                # Get predictions
                dominant_emotion = CLASS_NAMES[np.argmax(predictions)]
                mental_health = MENTAL_HEALTH_MAPPING[dominant_emotion]
                
                # Display dominant emotion with mental health context
                st.markdown(f"""
                <div class="wellness-tip">
                    <h3 style='color: {mental_health["color"]};'>Primary Emotion: {dominant_emotion.upper()}</h3>
                    <p><strong>Status:</strong> {mental_health['status']}</p>
                    <p><strong>Wellness Insight:</strong> {mental_health['wellness_tip']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # All emotions chart
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
                
                # Recommended activities
                st.markdown("### 🌱 Recommended Activities")
                cols = st.columns(3)
                for i, activity in enumerate(mental_health['activities']):
                    with cols[i]:
                        if st.button(f"✅ {activity}", key=f"activity_{i}"):
                            st.success(f"Great! Try to {activity.lower()} now.")
                
                # Save to history
                if st.button("💾 Save to History", use_container_width=True):
                    analysis_data = {
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

def show_mental_health_tips():
    st.markdown("## 💚 Mental Health Tips")
    st.markdown("Personalized wellness tips based on your emotional state")
    
    # Filter by emotion
    selected_emotion = st.selectbox(
        "Filter tips by emotion",
        ["All"] + [e.capitalize() for e in CLASS_NAMES]
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

def show_history():
    st.markdown("## 📊 Your Emotional Journey")
    st.markdown("Track your emotional patterns over time")
    
    if not st.session_state.analysis_history:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <img src="https://img.icons8.com/fluency/240/000000/sad.png" 
                     style="width: 120px; height: auto; opacity: 0.5;">
                <h3 style="color: #666;">No Analysis History Yet</h3>
                <p>Start by analyzing your emotions using the Emotion Analysis page!</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Convert to DataFrame for analysis with safe extraction
    history_data = []
    for entry in st.session_state.analysis_history:
        if isinstance(entry, dict):
            # Handle if entry itself is the data or contains 'data' key
            if 'dominant_emotion' in entry:
                history_data.append(entry)
            elif 'data' in entry and isinstance(entry['data'], dict):
                data_entry = entry['data'].copy()
                if 'timestamp' not in data_entry and 'timestamp' in entry:
                    data_entry['timestamp'] = entry['timestamp']
                history_data.append(data_entry)
    
    if not history_data:
        st.warning("No valid history data found.")
        return
    
    df = pd.DataFrame(history_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyses", len(df))
    with col2:
        most_common = df['dominant_emotion'].mode()[0] if not df.empty and len(df['dominant_emotion'].mode()) > 0 else "N/A"
        st.metric("Most Common Emotion", most_common.capitalize() if most_common != "N/A" else "N/A")
    with col3:
        avg_confidence = df['confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    with col4:
        unique_emotions = df['dominant_emotion'].nunique()
        st.metric("Unique Emotions", unique_emotions)
    
    st.markdown("---")
    
    # Emotion distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎭 Emotion Distribution")
        emotion_counts = df['dominant_emotion'].value_counts()
        if not emotion_counts.empty:
            fig_pie = px.pie(
                values=emotion_counts.values,
                names=[e.capitalize() for e in emotion_counts.index],
                title="Your Emotional Patterns",
                color_discrete_sequence=[MENTAL_HEALTH_MAPPING.get(e, {}).get('color', '#808080') for e in emotion_counts.index]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No emotion data to display")
    
    with col2:
        st.markdown("### 📊 Emotion Frequency")
        emotion_trends = df['dominant_emotion'].value_counts().head(5)
        if not emotion_trends.empty:
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=[e.capitalize() for e in emotion_trends.index],
                    y=emotion_trends.values,
                    marker_color=[MENTAL_HEALTH_MAPPING.get(e, {}).get('color', '#808080') for e in emotion_trends.index],
                    text=emotion_trends.values,
                    textposition='auto',
                )
            ])
            fig_bar.update_layout(
                title="Top 5 Emotions",
                xaxis_title="Emotion",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No frequency data to display")
    
    # Timeline view
    st.markdown("### 📈 Emotion Timeline")
    fig_timeline = create_emotion_timeline(df)
    if fig_timeline:
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Recent history
    st.markdown("### 📋 Recent Analyses")
    for i, entry in enumerate(reversed(history_data[-10:])):
        emotion = entry['dominant_emotion']
        color = MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', '#808080')
        timestamp = entry.get('timestamp', 'Unknown time')
        confidence = entry.get('confidence', 0) * 100
        with st.container():
            st.markdown(f"""
            <div class="history-entry" style="border-left: 5px solid {color};">
                <strong>Analysis #{len(history_data) - i}</strong><br>
                Time: {timestamp}<br>
                Emotion: <span style="color:{color}; font-weight:bold;">{emotion.upper()}</span><br>
                Confidence: {confidence:.1f}%
            </div>
            """, unsafe_allow_html=True)

# ==================== PROFESSIONAL PDF REPORT GENERATION ====================
class ProfessionalPDF(FPDF):
    """Enhanced PDF class for professional hospital-style reports"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)
        
    def header(self):
        # Logo and header
        self.set_fill_color(0, 94, 184)  # Hospital blue
        self.rect(0, 0, 210, 40, 'F')
        
        # Title
        self.set_y(10)
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'NEUROWELL MENTAL HEALTH CENTER', 0, 1, 'C')
        
        # Subtitle
        self.set_font('Helvetica', '', 12)
        self.set_text_color(230, 230, 230)
        self.cell(0, 8, 'Emotional Wellness Assessment Report', 0, 1, 'C')
        
        # Line
        self.set_y(42)
        self.set_draw_color(255, 184, 28)  # Warm yellow
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
        
        # Replace special characters with safe alternatives
        replacements = {
            '•': '-',           # Bullet point
            '●': '-',           # Black circle
            '○': 'o',           # White circle
            '◆': '*',           # Diamond
            '▪': '-',           # Small square
            '✓': '+',           # Check mark
            '✗': 'X',           # X mark
            '★': '*',           # Star
            '☆': '*',           # Star outline
            '♥': '<3',          # Heart
            '♦': '<>',          # Diamond
            '♣': 'C',           # Club
            '♠': 'S',           # Spade
            '→': '->',          # Right arrow
            '←': '<-',          # Left arrow
            '↑': '^',           # Up arrow
            '↓': 'v',           # Down arrow
            '©': '(C)',         # Copyright
            '®': '(R)',         # Registered
            '™': '(TM)',        # Trademark
            '°': ' degrees',    # Degree
            '±': '+/-',         # Plus-minus
            '×': 'x',           # Multiplication
            '÷': '/',           # Division
            '€': 'EUR',         # Euro
            '£': 'GBP',         # Pound
            '¥': 'JPY',         # Yen
            '₹': 'INR',         # Rupee
            '«': '"',           # Left quote
            '»': '"',           # Right quote
            '—': '-',           # Em dash
            '–': '-',           # En dash
            '…': '...',         # Ellipsis
            '“': '"',           # Left double quote
            '”': '"',           # Right double quote
            '‘': "'",           # Left single quote
            '’': "'",           # Right single quote
            '\u2022': '-',      # Bullet point (unicode)
            '\u2023': '-',      # Triangular bullet
            '\u25CF': '-',      # Black circle
            '\u25CB': 'o',      # White circle
            '\u25A0': '-',      # Black square
            '\u25A1': '[]',     # White square
        }
        
        # Replace emojis with text representations
        emoji_replacements = {
            '😊': ':happy:',
            '😠': ':angry:',
            '😢': ':sad:',
            '😨': ':fear:',
            '😐': ':neutral:',
            '😲': ':surprise:',
            '🤢': ':disgust:',
            '😤': ':frustrated:',
            '😰': ':anxious:',
            '😔': ':sad:',
            '😱': ':terrified:',
            '😡': ':furious:',
            '🥺': ':pleading:',
            '🤔': ':thoughtful:',
            '😌': ':relieved:',
            '😎': ':cool:',
            '🥳': ':celebrating:',
            '😴': ':sleepy:',
            '🤒': ':sick:',
            '🤕': ':hurt:',
            '🥴': ':dizzy:',
            '🤯': ':mind-blown:',
            '😬': ':nervous:',
            '🥲': ':tearful:',
            '😶‍🌫️': ':foggy:',
            '😮‍💨': ':exhaling:',
            '😵‍💫': ':dizzy:',
            '😳': ':embarrassed:',
        }
        
        # Apply replacements
        for old, new in {**replacements, **emoji_replacements}.items():
            text = text.replace(old, new)
        
        # Remove any remaining non-ASCII characters
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
        # Table header
        self.set_fill_color(0, 94, 184)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 10)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, self._clean_text(header), 1, 0, 'C', 1)
        self.ln()
        
        # Table data
        self.set_text_color(0, 0, 0)
        self.set_font('Helvetica', '', 9)
        fill = False
        
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 8, self._clean_text(str(item)), 'LR', 0, 'L' if i == 0 else 'C', fill)
            self.ln()
            fill = not fill
        
        # Closing line
        self.cell(sum(col_widths), 0, '', 'T')

def generate_professional_pdf_report(df, username, start_date, end_date, title):
    """Generate a professional hospital-style PDF report"""
    
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
    positive_count = len(df[df['dominant_emotion'].isin(
        [e for e in CLASS_NAMES if MENTAL_HEALTH_MAPPING.get(e, {}).get('category') == 'Positive']
    )])
    negative_count = len(df[df['dominant_emotion'].isin(
        [e for e in CLASS_NAMES if MENTAL_HEALTH_MAPPING.get(e, {}).get('category') == 'Negative']
    )])
    neutral_count = len(df[df['dominant_emotion'].isin(
        [e for e in CLASS_NAMES if MENTAL_HEALTH_MAPPING.get(e, {}).get('category') == 'Neutral']
    )])
    avg_confidence = df['confidence'].mean() * 100
    emotional_diversity = df['dominant_emotion'].nunique()
    most_common = df['dominant_emotion'].mode()[0] if not df.empty else "neutral"
    
    # Wellness score calculation
    positive_ratio = positive_count / len(df) if len(df) > 0 else 0
    negative_ratio = negative_count / len(df) if len(df) > 0 else 0
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
    pdf.set_text_color(66, 85, 99)  # Slate grey
    
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
    pdf.ln(15)
    
    # Recent Activity
    pdf.section_title('RECENT ASSESSMENT LOG')
    
    headers = ['Date/Time', 'Emotion', 'Confidence', 'Status']
    col_widths = [45, 30, 30, 45]
    
    recent_data = []
    recent_df = df.sort_values('datetime', ascending=False).head(8)
    for _, row in recent_df.iterrows():
        timestamp = row['datetime'].strftime('%Y-%m-%d %H:%M') if hasattr(row['datetime'], 'strftime') else str(row['timestamp'])[:16]
        emotion = row['dominant_emotion'].capitalize()
        confidence = f"{row['confidence']*100:.1f}%"
        status = MENTAL_HEALTH_MAPPING.get(row['dominant_emotion'], {}).get('status', '')
        # Clean the status text
        status = pdf._clean_text(status)[:20]
        recent_data.append([timestamp, emotion, confidence, status])
    
    pdf.create_table(headers, recent_data, col_widths)
    pdf.ln(15)
    
    # Treatment Recommendations - WITHOUT using bullet points
    pdf.section_title('TREATMENT RECOMMENDATIONS')
    
    most_common_data = MENTAL_HEALTH_MAPPING.get(most_common, MENTAL_HEALTH_MAPPING['neutral'])
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(0, 94, 184)
    pdf.cell(0, 8, f'Primary Recommendation (based on {most_common.capitalize()} predominance):', 0, 1)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(66, 85, 99)
    
    # Clean activities text - replace bullet points with hyphens
    activities = [pdf._clean_text(act) for act in most_common_data["activities"]]
    intervention = pdf._clean_text(most_common_data.get("intervention", "Cognitive behavioral therapy"))
    
    # Use hyphens instead of bullet points
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
    # Use numbers instead of bullet points
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
    
    # Return PDF as bytes with proper encoding
    return pdf.output(dest='S').encode('latin-1', errors='ignore')

def show_analytics():
    st.markdown(f"## 📈 Advanced Emotion Analytics")
    st.markdown(f"Deep dive into your emotional patterns with multiple visualization types")
    
    if not st.session_state.analysis_history:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem;">
                <img src="https://img.icons8.com/fluency/240/000000/sad.png" 
                     style="width: 120px; height: auto; opacity: 0.5;">
                <h3 style="color: {COLORS['gray']};">No Data Available</h3>
                <p style="color: {COLORS['gray']};">Complete some emotion analyses first to see your analytics!</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    history_data = []
    for entry in st.session_state.analysis_history:
        if isinstance(entry, dict):
            if 'dominant_emotion' in entry:
                history_data.append(entry)
            elif 'data' in entry and isinstance(entry['data'], dict):
                data_entry = entry['data'].copy()
                if 'timestamp' not in data_entry and 'timestamp' in entry:
                    data_entry['timestamp'] = entry['timestamp']
                history_data.append(data_entry)
    
    df = pd.DataFrame(history_data)
    
    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            help="Select the start date for analysis"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="Select the end date for analysis"
        )
    
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        mask = (df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        st.warning("No data available for the selected date range.")
        return
    
    st.markdown("### 📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    positive_count = len(df_filtered[df_filtered['dominant_emotion'].isin(
        [e for e in CLASS_NAMES if MENTAL_HEALTH_MAPPING.get(e, {}).get('category') == 'Positive']
    )])
    negative_count = len(df_filtered[df_filtered['dominant_emotion'].isin(
        [e for e in CLASS_NAMES if MENTAL_HEALTH_MAPPING.get(e, {}).get('category') == 'Negative']
    )])
    avg_confidence = df_filtered['confidence'].mean() * 100
    unique_days = df_filtered['datetime'].dt.date.nunique() if 'datetime' in df_filtered.columns else 1
    
    with col1:
        st.metric("Positive Moments", positive_count)
    with col2:
        st.metric("Challenging Moments", negative_count)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    with col4:
        st.metric("Active Days", unique_days)
    
    chart_type = st.radio(
        "Select Chart Type",
        ["Timeline", "Radar Profile", "Sunburst", "Treemap", "Heatmap", "Weekly Pattern", "Transition Flow"],
        horizontal=True,
        help="Choose different visualization types to explore your data"
    )
    
    if chart_type == "Timeline":
        fig = create_emotion_timeline(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.info("📝 **Insight:** This timeline shows how your emotions change over time.")
    
    elif chart_type == "Radar Profile":
        fig = create_emotion_radar(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.info("📝 **Insight:** Your emotion radar shows the intensity of different emotions.")
    
    elif chart_type == "Sunburst":
        fig = create_emotion_sunburst(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.info("📝 **Insight:** The sunburst chart shows the hierarchy of your emotions.")
    
    elif chart_type == "Treemap":
        fig = create_emotion_treemap(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.info("📝 **Insight:** This treemap breaks down emotions by confidence levels.")
    
    elif chart_type == "Heatmap":
        figs = create_emotion_heatmap(df_filtered)
        if figs:
            tabs = st.tabs([e.capitalize() for e in CLASS_NAMES if e in df_filtered['dominant_emotion'].unique()])
            for i, (fig, tab) in enumerate(zip(figs, tabs)):
                with tab:
                    st.plotly_chart(fig, use_container_width=True)
            st.info("📝 **Insight:** Heatmaps show when you're most likely to feel each emotion.")
    
    elif chart_type == "Weekly Pattern":
        fig = create_weekly_pattern(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.info("📝 **Insight:** This chart shows your emotional patterns by day.")
    
    elif chart_type == "Transition Flow":
        fig = create_emotion_flow(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.info("📝 **Insight:** The flow diagram shows how your emotions transition.")
    
    # Professional PDF Export Section
    st.markdown("### 📋 Clinical Report Generation")
    st.markdown("Generate a comprehensive clinical report in hospital format")
    
    col1, col2 = st.columns(2)
    with col1:
        report_title = st.text_input("Report Title", value=f"NeuroWell Clinical Assessment - {st.session_state.username}")
    with col2:
        st.markdown("### Report Type")
        st.markdown("**🏥 Comprehensive Clinical Assessment**")
    
    if st.button("📄 Generate Hospital-Grade PDF Report", use_container_width=True, type="primary"):
        with st.spinner("Generating professional clinical report..."):
            try:
                pdf_bytes = generate_professional_pdf_report(
                    df_filtered, 
                    st.session_state.username,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    report_title
                )
                
                st.download_button(
                    label="📥 Download Clinical Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"NeuroWell_Clinical_Report_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                st.success("✅ Clinical PDF report generated successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                st.info("Please ensure fpdf is installed: pip install fpdf")

def show_profile():
    st.markdown("## 👤 Your Profile")
    
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
        
        if st.session_state.username in users:
            user_data = users[st.session_state.username]
            
            # Profile Image Section
            st.markdown("### 📸 Profile Picture")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display current profile image
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
                    # Display preview
                    st.image(uploaded_file, caption="Preview", width=150)
                    
                    if st.button("💾 Save Profile Picture", use_container_width=True):
                        # Save the image
                        image_data = uploaded_file.getvalue()
                        success, result = save_profile_image(st.session_state.username, image_data)
                        
                        if success:
                            st.session_state.profile_image = result
                            st.success("✅ Profile picture updated successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error saving image: {result}")
            
            st.markdown("---")
            
            # Account Information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Account Information")
                st.info(f"**Username:** {st.session_state.username}")
                st.info(f"**Email:** {user_data['email']}")
                st.info(f"**Member since:** {user_data['created_at']}")
                st.info(f"**Total analyses:** {len(user_data.get('history', []))}")
            
            with col2:
                st.markdown("### Settings")
                # Load current settings or use defaults
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
            
            # Data Management
            st.markdown("### Data Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export My Data", use_container_width=True):
                    # Create export data
                    export_data = {
                        'username': st.session_state.username,
                        'email': user_data['email'],
                        'created_at': user_data['created_at'],
                        'settings': user_data.get('settings', {}),
                        'analysis_history': user_data.get('history', [])
                    }
                    
                    # Convert to JSON and offer download
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
            
            # Delete Account (with confirmation)
            st.markdown("### ⚠️ Danger Zone")
            with st.expander("Delete Account"):
                st.warning("This action cannot be undone. All your data will be permanently deleted.")
                confirm = st.text_input("Type 'DELETE' to confirm")
                if st.button("🗑️ Permanently Delete Account", use_container_width=True, type="primary"):
                    if confirm == "DELETE":
                        # Delete user data
                        del users[st.session_state.username]
                        with open(USER_DB_FILE, 'w') as f:
                            json.dump(users, f, indent=4)
                        
                        # Delete profile image
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

# ==================== MAIN EXECUTION ====================
def main():
    if not model:
        st.error("⚠️ Model file 'emotiondetector.h5' not found. Please ensure the model file exists in the project directory.")
        st.info("If you haven't trained the model yet, you need to train it first.")
        return
    
    if not st.session_state.logged_in:
        show_login_signup()
    else:
        show_main_app()

if __name__ == "__main__":
    main()