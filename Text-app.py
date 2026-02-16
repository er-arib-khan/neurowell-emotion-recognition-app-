import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import io
import base64
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Advanced Clinical Text Analysis System",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Enhanced sentiment analysis functions
class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.medical_terms = self.load_medical_terms()
        self.clinical_phrases = self.load_clinical_phrases()
        
    def load_medical_terms(self):
        """Load medical terminology for specialized analysis"""
        return {
            'positive': ['improved', 'better', 'recovered', 'stable', 'normal', 'healthy', 
                        'responding', 'progress', 'good', 'excellent', 'clear', 'negative_test',
                        'benign', 'remission', 'improving', 'successful', 'favorable'],
            'negative': ['worsening', 'deteriorating', 'severe', 'critical', 'abnormal', 
                        'positive_test', 'infection', 'complication', 'failure', 'rejection',
                        'malignant', 'metastasis', 'acute', 'chronic', 'progressive'],
            'uncertain': ['possible', 'maybe', 'suspected', 'rule_out', 'differential', 
                         'uncertain', 'unclear', 'equivocal', 'probable', 'questionable',
                         'rule out', 'consider', '?']
        }
    
    def load_clinical_phrases(self):
        """Load clinical phrases and their sentiment weights"""
        return {
            'vital_signs_stable': 0.3,
            'shows_improvement': 0.5,
            'no_significant_change': 0.0,
            'requires_monitoring': -0.1,
            'critical_condition': -0.8,
            'emergency_situation': -0.9,
            'positive_response': 0.6,
            'adverse_reaction': -0.7,
            'successful_procedure': 0.7,
            'postoperative_complication': -0.6,
            'patient_stable': 0.4,
            'guarded_condition': -0.3,
            'poor_prognosis': -0.7,
            'excellent_prognosis': 0.8,
            'routine_followup': 0.2
        }
    
    def analyze_sentiment_vader(self, text):
        """VADER sentiment analysis - good for social media and general text"""
        return self.sia.polarity_scores(text)
    
    def analyze_sentiment_textblob(self, text):
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_medical_sentiment(self, text):
        """Specialized medical sentiment analysis"""
        text_lower = text.lower()
        words = text_lower.split()
        
        medical_score = 0
        medical_terms_found = {'positive': [], 'negative': [], 'uncertain': [], 'clinical': []}
        
        # Check for medical terms
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in text_lower:
                    medical_terms_found[category].append(term)
                    if category == 'positive':
                        medical_score += 0.2
                    elif category == 'negative':
                        medical_score -= 0.3
                    elif category == 'uncertain':
                        medical_score -= 0.1
        
        # Check for clinical phrases
        for phrase, weight in self.clinical_phrases.items():
            phrase_text = phrase.replace('_', ' ')
            if phrase_text in text_lower:
                medical_score += weight
                medical_terms_found['clinical'].append(phrase_text)
        
        # Check for negations (important in medical context)
        negation_words = ['no', 'not', 'without', 'denies', 'absence of', 'negative for']
        for negation in negation_words:
            if negation in text_lower:
                # Find words after negation and adjust sentiment
                negation_index = text_lower.find(negation)
                if negation_index != -1:
                    medical_score *= -0.5  # Reverse and reduce sentiment
        
        # Normalize score to [-1, 1]
        medical_score = max(-1, min(1, medical_score))
        
        return {
            'medical_score': medical_score,
            'medical_terms_found': medical_terms_found
        }
    
    def analyze_emotions_advanced(self, text):
        """Advanced emotion detection with weighted scoring"""
        emotion_patterns = {
            'joy': {
                'words': ['happy', 'joy', 'delighted', 'pleased', 'grateful', 'thankful', 
                         'hopeful', 'optimistic', 'positive', 'good', 'great', 'excellent',
                         'wonderful', 'fantastic', 'amazing', 'beautiful', 'relieved',
                         'encouraged', 'encouraging'],
                'clinical': ['responding well', 'improving', 'recovering', 'stable condition']
            },
            'sadness': {
                'words': ['sad', 'depressed', 'unhappy', 'miserable', 'gloomy', 'hopeless',
                         'devastated', 'heartbroken', 'grief', 'mourning', 'distressed',
                         'tearful', 'crying', 'sorrow', 'despair'],
                'clinical': ['poor prognosis', 'terminal', 'end stage', 'palliative']
            },
            'anger': {
                'words': ['angry', 'mad', 'furious', 'frustrated', 'irritated', 'annoyed',
                         'hostile', 'aggressive', 'outraged', 'resentful', 'bitter',
                         'frustration', 'irritation'],
                'clinical': ['combative', 'agitated', 'refusing treatment', 'non-compliant']
            },
            'fear': {
                'words': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous',
                         'panicked', 'frightened', 'alarmed', 'dread', 'fearful',
                         'anxiety', 'panic', 'phobia'],
                'clinical': ['fearful', 'anxious about procedure', 'needle phobia', 'health anxiety']
            },
            'surprise': {
                'words': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned',
                         'unexpected', 'unanticipated', 'sudden', 'startled',
                         'remarkable', 'extraordinary'],
                'clinical': ['unexpected finding', 'surprising diagnosis', 'sudden change']
            },
            'trust': {
                'words': ['trust', 'confident', 'believe', 'faith', 'reliable', 'dependable',
                         'assured', 'certain', 'sure', 'confidence', 'trusting',
                         'trustworthy'],
                'clinical': ['trust in treatment', 'confident in recovery', 'reliable improvement']
            },
            'anticipation': {
                'words': ['expect', 'anticipate', 'await', 'look forward', 'prospective',
                         'pending', 'upcoming', 'future', 'expecting', 'awaiting',
                         'preparing'],
                'clinical': ['awaiting results', 'pending tests', 'scheduled procedure']
            }
        }
        
        text_lower = text.lower()
        emotions_detected = {}
        
        for emotion, patterns in emotion_patterns.items():
            score = 0
            
            # Check regular words
            for word in patterns['words']:
                count = len(re.findall(r'\b' + word + r'\b', text_lower))
                if count > 0:
                    score += count * 0.3  # Base weight for word matches
            
            # Check clinical phrases (higher weight)
            for phrase in patterns.get('clinical', []):
                if phrase in text_lower:
                    score += 0.5  # Higher weight for clinical phrases
            
            # Check for intensifiers
            intensifiers = ['very', 'extremely', 'severely', 'highly', 'intensely', 'particularly']
            for intensifier in intensifiers:
                for word in patterns['words']:
                    if f"{intensifier} {word}" in text_lower:
                        score += 0.3  # Additional weight for intensified emotions
            
            if score > 0:
                emotions_detected[emotion] = round(score, 2)
        
        return emotions_detected
    
    def analyze_clinical_urgency(self, text):
        """Detect clinical urgency and priority"""
        urgency_indicators = {
            'critical': ['emergency', 'critical', 'life-threatening', 'code blue', 'resuscitate',
                        'immediate', 'stat', 'urgent', 'acute', 'severe', 'crashing',
                        'unresponsive', 'intubate', 'cardiac arrest', 'respiratory failure'],
            'high': ['severe', 'serious', 'significant', 'major', 'intensive', 'unstable',
                    'deteriorating', 'worsening', 'rapid', 'aggressive'],
            'medium': ['moderate', 'concerning', 'worsening', 'deteriorating', 'guarded',
                      'monitor', 'observation', 'pending'],
            'low': ['mild', 'minor', 'stable', 'routine', 'follow-up', 'scheduled',
                   'elective', 'non-urgent']
        }
        
        text_lower = text.lower()
        urgency_scores = {level: 0 for level in urgency_indicators}
        
        for level, indicators in urgency_indicators.items():
            for indicator in indicators:
                count = len(re.findall(r'\b' + indicator + r'\b', text_lower))
                if count > 0:
                    urgency_scores[level] += count
        
        # Determine overall urgency with weighted scoring
        total_score = (
            urgency_scores['critical'] * 4 +
            urgency_scores['high'] * 3 +
            urgency_scores['medium'] * 2 +
            urgency_scores['low'] * 1
        )
        
        if urgency_scores['critical'] > 0:
            overall = 'CRITICAL'
        elif urgency_scores['high'] > 0 or total_score >= 10:
            overall = 'HIGH'
        elif urgency_scores['medium'] > 0 or total_score >= 5:
            overall = 'MEDIUM'
        elif urgency_scores['low'] > 0 or total_score > 0:
            overall = 'LOW'
        else:
            overall = 'ROUTINE'
        
        return {
            'level': overall,
            'scores': urgency_scores,
            'total_score': total_score
        }
    
    def analyze_readability(self, text):
        """Simple readability analysis without external libraries"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Simple syllable count estimation
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = 'aeiouy'
            if word and word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index-1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') and len(word) > 2:
                count += 1
            if count == 0:
                count += 1
            return count
        
        total_syllables = sum(count_syllables(word) for word in words)
        
        # Flesch Reading Ease approximation
        if len(sentences) > 0 and len(words) > 0:
            flesch_score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
        else:
            flesch_score = 0
        
        return {
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'avg_word_length': round(avg_word_length, 1),
            'syllable_count': total_syllables,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    def extract_medical_entities(self, text):
        """Extract medical entities using pattern matching"""
        entities = {
            'symptoms': [],
            'diagnoses': [],
            'medications': [],
            'procedures': [],
            'body_parts': []
        }
        
        # Common medical indicators
        symptom_keywords = ['pain', 'ache', 'discomfort', 'swelling', 'fever', 'cough', 
                           'fatigue', 'nausea', 'vomiting', 'dizziness', 'headache',
                           'rash', 'bleeding', 'infection', 'inflammation']
        
        diagnosis_keywords = ['diagnosed with', 'suffering from', 'presented with',
                             'history of', 'complains of', 'indicates']
        
        medication_keywords = ['mg', 'tablet', 'capsule', 'injection', 'prescribed',
                              'medication', 'drug', 'dose', 'treatment']
        
        procedure_keywords = ['surgery', 'procedure', 'operation', 'biopsy', 'scan', 
                             'x-ray', 'mri', 'ct', 'ultrasound', 'endoscopy',
                             'colonoscopy', 'mammogram']
        
        body_parts = ['heart', 'lung', 'brain', 'liver', 'kidney', 'stomach', 'spine',
                     'joint', 'bone', 'muscle', 'skin', 'blood', 'chest', 'abdomen',
                     'head', 'neck', 'back', 'arm', 'leg', 'foot', 'hand']
        
        text_lower = text.lower()
        
        # Extract symptoms
        for symptom in symptom_keywords:
            if symptom in text_lower:
                entities['symptoms'].append(symptom)
        
        # Extract body parts
        for body_part in body_parts:
            if body_part in text_lower:
                entities['body_parts'].append(body_part)
        
        # Look for medications (simplified - looks for common medication patterns)
        medication_pattern = r'\b\d+\s*(mg|mcg|g|ml|tablet|capsule)\b'
        medications = re.findall(medication_pattern, text_lower)
        if medications:
            entities['medications'].extend(medications)
        
        # Look for procedures
        for procedure in procedure_keywords:
            if procedure in text_lower:
                entities['procedures'].append(procedure)
        
        return entities

# Initialize the advanced analyzer
analyzer = AdvancedSentimentAnalyzer()

# PDF Report Generation Functions
def generate_pdf_report(analysis):
    """Generate a PDF report for a single analysis"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    story.append(Paragraph("Advanced Clinical Text Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Report Info
    story.append(Paragraph(f"Date: {analysis['timestamp']}", styles['Normal']))
    if analysis.get('patient_id'):
        story.append(Paragraph(f"Patient ID: {analysis['patient_id']}", styles['Normal']))
    if analysis.get('doctor_name'):
        story.append(Paragraph(f"Doctor: {analysis['doctor_name']}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Text Sample
    story.append(Paragraph("Analyzed Text:", styles['Heading2']))
    story.append(Paragraph(analysis['text'], styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Sentiment Analysis Results
    story.append(Paragraph("Sentiment Analysis Results:", styles['Heading2']))
    
    data = [
        ['Metric', 'Value'],
        ['Overall Sentiment', analysis['sentiment']['category']],
        ['Polarity Score', f"{analysis['sentiment']['polarity']:.3f}"],
        ['Subjectivity Score', f"{analysis['sentiment']['subjectivity']:.3f}"],
        ['Medical Sentiment Score', f"{analysis['advanced_sentiment']['medical_score']:.3f}"],
        ['Clinical Urgency', analysis['advanced_sentiment']['urgency']['level']],
        ['Word Count', str(analysis['statistics']['word_count'])],
        ['Character Count', str(analysis['statistics']['char_count'])],
        ['Sentence Count', str(analysis['statistics']['sentence_count'])]
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Emotions Detected
    if analysis['emotions']:
        story.append(Paragraph("Emotions Detected:", styles['Heading2']))
        emotion_data = [['Emotion', 'Intensity']]
        for emotion, intensity in analysis['emotions'].items():
            emotion_data.append([emotion.capitalize(), str(intensity)])
        
        emotion_table = Table(emotion_data)
        emotion_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(emotion_table)
        story.append(Spacer(1, 20))
    
    # Medical Terms Found
    if analysis['advanced_sentiment']['medical_terms_found']:
        story.append(Paragraph("Medical Terms Detected:", styles['Heading2']))
        medical_data = [['Category', 'Terms']]
        for category, terms in analysis['advanced_sentiment']['medical_terms_found'].items():
            if terms:
                medical_data.append([category.capitalize(), ', '.join(terms[:5])])
        
        if len(medical_data) > 1:
            medical_table = Table(medical_data)
            medical_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(medical_table)
            story.append(Spacer(1, 20))
    
    # Key Phrases
    if analysis['key_phrases']:
        story.append(Paragraph("Key Phrases Detected:", styles['Heading2']))
        for phrase in analysis['key_phrases']:
            story.append(Paragraph(f"• {phrase}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Word Cloud
    if 'wordcloud_img' in analysis and analysis['wordcloud_img']:
        try:
            story.append(Paragraph("Word Cloud Visualization:", styles['Heading2']))
            img = Image(analysis['wordcloud_img'], width=6*inch, height=3*inch)
            story.append(img)
        except:
            story.append(Paragraph("Word cloud image could not be generated", styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("Generated by Advanced Clinical Text Analysis System", styles['Italic']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Download PDF
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="advanced_clinical_analysis_report.pdf">📥 Download Advanced PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)

def generate_comprehensive_report(df, start_date, end_date):
    """Generate a comprehensive PDF report with statistics"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    story.append(Paragraph("Comprehensive Clinical Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Date Range
    story.append(Paragraph(f"Report Period: {start_date} to {end_date}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Summary Statistics
    story.append(Paragraph("Summary Statistics:", styles['Heading2']))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Analyses', str(len(df))],
        ['Average Polarity', f"{df['Polarity'].mean():.3f}"],
        ['Positive Analyses', str((df['Sentiment'] == 'Positive').sum())],
        ['Negative Analyses', str((df['Sentiment'] == 'Negative').sum())],
        ['Neutral Analyses', str((df['Sentiment'] == 'Neutral').sum())],
        ['Average Word Count', f"{df['Word Count'].mean():.0f}"]
    ]
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Sentiment Distribution
    story.append(Paragraph("Sentiment Distribution:", styles['Heading2']))
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_data = [['Sentiment', 'Count', 'Percentage']]
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        sentiment_data.append([sentiment, str(count), f"{percentage:.1f}%"])
    
    sentiment_table = Table(sentiment_data)
    sentiment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(sentiment_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Download PDF
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="comprehensive_report.pdf">📥 Download Comprehensive Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .report-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .urgency-critical {
        color: #dc3545;
        font-weight: bold;
        background-color: #f8d7da;
        padding: 5px;
        border-radius: 5px;
    }
    .urgency-high {
        color: #fd7e14;
        font-weight: bold;
        background-color: #fff3cd;
        padding: 5px;
        border-radius: 5px;
    }
    .urgency-medium {
        color: #ffc107;
        font-weight: bold;
        background-color: #fff3cd;
        padding: 5px;
        border-radius: 5px;
    }
    .urgency-low {
        color: #17a2b8;
        font-weight: bold;
        background-color: #d1ecf1;
        padding: 5px;
        border-radius: 5px;
    }
    .urgency-routine {
        color: #28a745;
        font-weight: bold;
        background-color: #d4edda;
        padding: 5px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏥 Advanced Clinical Text Analysis System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio(
        "Select Page",
        ["Text Analysis", "History", "Reports", "Settings"]
    )
    
    st.markdown("---")
    st.header("📈 Statistics")
    
    if st.session_state.history:
        total_analyses = len(st.session_state.history)
        avg_sentiment = np.mean([h['sentiment']['polarity'] for h in st.session_state.history])
        avg_medical_score = np.mean([h['advanced_sentiment']['medical_score'] for h in st.session_state.history])
        
        st.metric("Total Analyses", total_analyses)
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        st.metric("Average Medical Score", f"{avg_medical_score:.2f}")
    else:
        st.info("No analyses yet")

# Text Analysis Page
if page == "Text Analysis":
    st.header("📝 Advanced Text Analysis Tool")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input text
        text_input = st.text_area(
            "Enter text for analysis:",
            height=200,
            placeholder="Paste or type clinical notes, patient feedback, or any text here..."
        )
        
        # Analysis options
        col_a, col_b = st.columns(2)
        with col_a:
            analyze_btn = st.button("🔍 Analyze Text", use_container_width=True)
        with col_b:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
            
        if clear_btn:
            text_input = ""
            st.rerun()
    
    with col2:
        st.subheader("⚙️ Analysis Settings")
        include_wordcloud = st.checkbox("Include Word Cloud", value=True)
        include_emotions = st.checkbox("Detect Emotions", value=True)
        include_medical_terms = st.checkbox("Detect Medical Terms", value=True)
        include_urgency = st.checkbox("Assess Clinical Urgency", value=True)
        save_to_history = st.checkbox("Save to History", value=True)
        
        st.subheader("📋 Patient Info (Optional)")
        patient_id = st.text_input("Patient ID")
        doctor_name = st.text_input("Doctor Name")
    
    if analyze_btn and text_input:
        with st.spinner("Performing advanced text analysis..."):
            # Basic sentiment analysis
            blob = TextBlob(text_input)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
            
            # VADER sentiment analysis
            vader_scores = analyzer.analyze_sentiment_vader(text_input)
            
            # Medical sentiment analysis
            medical_analysis = analyzer.analyze_medical_sentiment(text_input)
            
            # Clinical urgency assessment
            urgency = analyzer.analyze_clinical_urgency(text_input)
            
            # Advanced emotion detection
            emotions_detected = analyzer.analyze_emotions_advanced(text_input)
            
            # Readability analysis
            readability = analyzer.analyze_readability(text_input)
            
            # Medical entity extraction
            medical_entities = analyzer.extract_medical_entities(text_input)
            
            # Combine sentiment scores for overall assessment
            combined_polarity = (
                sentiment_polarity * 0.3 + 
                vader_scores['compound'] * 0.3 + 
                medical_analysis['medical_score'] * 0.4
            )
            
            # Determine sentiment category with more nuanced thresholds
            if combined_polarity > 0.2:
                sentiment_category = "Positive"
                sentiment_color = "sentiment-positive"
            elif combined_polarity < -0.2:
                sentiment_category = "Negative"
                sentiment_color = "sentiment-negative"
            elif combined_polarity > 0.05:
                sentiment_category = "Mildly Positive"
                sentiment_color = "sentiment-positive"
            elif combined_polarity < -0.05:
                sentiment_category = "Mildly Negative"
                sentiment_color = "sentiment-negative"
            else:
                sentiment_category = "Neutral"
                sentiment_color = "sentiment-neutral"
            
            # Get word counts
            words = text_input.split()
            word_count = len(words)
            char_count = len(text_input)
            sentence_count = len(text_input.split('.'))
            
            # Extract key phrases using TF-IDF approach
            try:
                vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([text_input])
                feature_names = vectorizer.get_feature_names_out()
                key_phrases = list(feature_names)
            except:
                # Fallback to simple noun extraction
                nouns = [word for word, pos in blob.tags if pos.startswith('NN')]
                key_phrases = list(set(nouns[:5]))
            
            # Store results
            current_analysis = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'text': text_input[:200] + "..." if len(text_input) > 200 else text_input,
                'full_text': text_input,
                'sentiment': {
                    'polarity': combined_polarity,
                    'original_polarity': sentiment_polarity,
                    'vader_compound': vader_scores['compound'],
                    'subjectivity': sentiment_subjectivity,
                    'category': sentiment_category,
                    'color': sentiment_color
                },
                'advanced_sentiment': {
                    'medical_score': medical_analysis['medical_score'],
                    'medical_terms_found': medical_analysis['medical_terms_found'],
                    'urgency': urgency
                },
                'emotions': emotions_detected,
                'readability': readability,
                'medical_entities': medical_entities,
                'statistics': {
                    'word_count': word_count,
                    'char_count': char_count,
                    'sentence_count': sentence_count
                },
                'key_phrases': key_phrases[:10],
                'patient_id': patient_id if patient_id else None,
                'doctor_name': doctor_name if doctor_name else None
            }
            
            st.session_state.current_analysis = current_analysis
            
            if save_to_history:
                st.session_state.history.append(current_analysis)
    
    # Display results
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        
        st.markdown("---")
        st.header("📊 Advanced Analysis Results")
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Overall Sentiment", analysis['sentiment']['category'])
        with col2:
            st.metric("Sentiment Score", f"{analysis['sentiment']['polarity']:.2f}")
        with col3:
            st.metric("Medical Score", f"{analysis['advanced_sentiment']['medical_score']:.2f}")
        with col4:
            urgency_level = analysis['advanced_sentiment']['urgency']['level']
            urgency_class = f"urgency-{urgency_level.lower()}"
            st.markdown(f"**Clinical Urgency:** <span class='{urgency_class}'>{urgency_level}</span>", unsafe_allow_html=True)
        with col5:
            st.metric("Word Count", analysis['statistics']['word_count'])
        
        # Graphs and detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Analysis Comparison")
            
            # Create comparison chart
            sentiment_scores = {
                'TextBlob': analysis['sentiment']['original_polarity'],
                'VADER': analysis['sentiment']['vader_compound'],
                'Medical': analysis['advanced_sentiment']['medical_score'],
                'Combined': analysis['sentiment']['polarity']
            }
            
            fig = go.Figure(data=[
                go.Bar(name='Score', x=list(sentiment_scores.keys()), 
                       y=list(sentiment_scores.values()),
                       marker_color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
            ])
            fig.update_layout(title='Sentiment Analysis Methods Comparison',
                            yaxis_title='Sentiment Score',
                            yaxis_range=[-1, 1],
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if include_emotions and analysis['emotions']:
                st.subheader("Emotion Detection")
                
                # Emotion bar chart
                emotions_df = pd.DataFrame(
                    list(analysis['emotions'].items()),
                    columns=['Emotion', 'Intensity']
                ).sort_values('Intensity', ascending=True)
                
                fig = px.bar(
                    emotions_df,
                    x='Intensity',
                    y='Emotion',
                    orientation='h',
                    color='Intensity',
                    color_continuous_scale='Viridis',
                    title='Detected Emotions and Their Intensity'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Medical Terms Found
        if include_medical_terms:
            st.subheader("🏥 Medical Terms Detected")
            medical_terms = analysis['advanced_sentiment']['medical_terms_found']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if medical_terms.get('positive'):
                    st.markdown("**Positive Terms:**")
                    for term in medical_terms['positive'][:5]:
                        st.markdown(f"✓ {term}")
            
            with col2:
                if medical_terms.get('negative'):
                    st.markdown("**Negative Terms:**")
                    for term in medical_terms['negative'][:5]:
                        st.markdown(f"⚠ {term}")
            
            with col3:
                if medical_terms.get('uncertain'):
                    st.markdown("**Uncertain Terms:**")
                    for term in medical_terms['uncertain'][:5]:
                        st.markdown(f"? {term}")
        
        # Clinical Urgency Details
        if include_urgency:
            st.subheader("🚨 Clinical Urgency Assessment")
            urgency_scores = analysis['advanced_sentiment']['urgency']['scores']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Critical Indicators", urgency_scores.get('critical', 0))
            with col2:
                st.metric("High Priority", urgency_scores.get('high', 0))
            with col3:
                st.metric("Medium Priority", urgency_scores.get('medium', 0))
            with col4:
                st.metric("Low Priority", urgency_scores.get('low', 0))
        
        # Word Cloud
        if include_wordcloud and analysis['full_text']:
            st.subheader("☁️ Word Cloud")
            
            try:
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    stopwords=set(['patient', 'doctor', 'hospital', 'clinical', 'said', 'would'])
                ).generate(analysis['full_text'])
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
                # Save word cloud to buffer for PDF
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                analysis['wordcloud_img'] = img_buffer
                plt.close()
            except Exception as e:
                st.error(f"Could not generate word cloud: {str(e)}")
        
        # Key phrases and readability
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔑 Key Phrases")
            if analysis['key_phrases']:
                for phrase in analysis['key_phrases']:
                    st.markdown(f"- {phrase}")
            else:
                st.info("No key phrases detected")
        
        with col2:
            st.subheader("📖 Readability Analysis")
            if 'readability' in analysis:
                readability = analysis['readability']
                st.metric("Flesch Reading Ease", f"{readability.get('flesch_reading_ease', 0):.1f}")
                st.metric("Avg Sentence Length", f"{readability.get('avg_sentence_length', 0):.1f} words")
                st.metric("Avg Word Length", f"{readability.get('avg_word_length', 0):.1f} chars")
        
        # Generate Report Button
        if st.button("📄 Generate Advanced PDF Report", use_container_width=True):
            generate_pdf_report(analysis)

# History Page
elif page == "History":
    st.header("📋 Analysis History")
    
    if st.session_state.history:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_sentiment = st.selectbox(
                "Filter by Sentiment",
                ["All", "Positive", "Mildly Positive", "Neutral", "Mildly Negative", "Negative"]
            )
        with col2:
            filter_urgency = st.selectbox(
                "Filter by Urgency",
                ["All", "ROUTINE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            )
        with col3:
            search_term = st.text_input("Search in text", "")
        
        # Display history
        filtered_history = st.session_state.history.copy()
        
        if filter_sentiment != "All":
            filtered_history = [
                h for h in filtered_history 
                if h['sentiment']['category'] == filter_sentiment
            ]
        
        if filter_urgency != "All":
            filtered_history = [
                h for h in filtered_history 
                if h['advanced_sentiment']['urgency']['level'] == filter_urgency
            ]
        
        if search_term:
            filtered_history = [
                h for h in filtered_history 
                if search_term.lower() in h['text'].lower() or
                search_term.lower() in str(h.get('patient_id', '')).lower()
            ]
        
        for i, entry in enumerate(reversed(filtered_history)):
            urgency_level = entry['advanced_sentiment']['urgency']['level']
            urgency_class = f"urgency-{urgency_level.lower()}"
            
            with st.expander(f"Analysis {len(filtered_history)-i}: {entry['timestamp']} - Urgency: {urgency_level}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Text:** {entry['text']}")
                    st.markdown(f"**Sentiment:** :{entry['sentiment']['color']}[{entry['sentiment']['category']}] (Score: {entry['sentiment']['polarity']:.2f})")
                    st.markdown(f"**Medical Score:** {entry['advanced_sentiment']['medical_score']:.2f}")
                    st.markdown(f"**Clinical Urgency:** <span class='{urgency_class}'>{urgency_level}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Word Count:** {entry['statistics']['word_count']}")
                    
                    if entry.get('patient_id'):
                        st.markdown(f"**Patient ID:** {entry['patient_id']}")
                    if entry.get('doctor_name'):
                        st.markdown(f"**Doctor:** {entry['doctor_name']}")
                
                with col2:
                    if st.button(f"View Full Report", key=f"view_{i}"):
                        st.session_state.current_analysis = entry
                        st.rerun()
                    
                    if st.button(f"Delete", key=f"del_{i}"):
                        st.session_state.history.remove(entry)
                        st.rerun()
        
        # Export history
        st.markdown("---")
        if st.button("📥 Export History to CSV"):
            export_df = pd.DataFrame([
                {
                    'Timestamp': h['timestamp'],
                    'Sentiment': h['sentiment']['category'],
                    'Polarity': h['sentiment']['polarity'],
                    'Medical Score': h['advanced_sentiment']['medical_score'],
                    'Urgency': h['advanced_sentiment']['urgency']['level'],
                    'Word Count': h['statistics']['word_count'],
                    'Patient ID': h.get('patient_id', ''),
                    'Doctor': h.get('doctor_name', '')
                }
                for h in st.session_state.history
            ])
            
            csv = export_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="advanced_analysis_history.csv">📥 Download Advanced CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No analysis history yet. Perform some analyses first!")

# Reports Page
elif page == "Reports":
    st.header("📊 Generate Advanced Reports")
    
    if st.session_state.history:
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now().date())
        with col2:
            end_date = st.date_input("End Date", datetime.now().date())
        
        # Generate summary statistics with advanced metrics
        df = pd.DataFrame([
            {
                'Date': h['timestamp'].split()[0],
                'Sentiment': h['sentiment']['category'],
                'Polarity': h['sentiment']['polarity'],
                'Medical Score': h['advanced_sentiment']['medical_score'],
                'Urgency': h['advanced_sentiment']['urgency']['level'],
                'Word Count': h['statistics']['word_count']
            }
            for h in st.session_state.history
        ])
        
        # Filter by date
        df['Date'] = pd.to_datetime(df['Date'])
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df[mask]
        
        if not filtered_df.empty:
            # Create tabs for different report views
            tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Detailed Analysis", "📋 Raw Data"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Distribution")
                    fig = px.pie(
                        filtered_df,
                        names='Sentiment',
                        title='Sentiment Distribution',
                        color_discrete_map={'Positive': 'lightgreen', 
                                          'Mildly Positive': 'palegreen',
                                          'Neutral': 'lightgray', 
                                          'Mildly Negative': 'lightsalmon',
                                          'Negative': 'lightcoral'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Urgency Distribution")
                    urgency_counts = filtered_df['Urgency'].value_counts()
                    fig = px.bar(
                        x=urgency_counts.index,
                        y=urgency_counts.values,
                        title='Clinical Urgency Distribution',
                        color=urgency_counts.index,
                        color_discrete_map={'ROUTINE': 'lightgreen',
                                          'LOW': 'lightblue',
                                          'MEDIUM': 'orange',
                                          'HIGH': 'red',
                                          'CRITICAL': 'darkred'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Sentiment and Medical Score Trends")
                # Group by date for trends
                daily_avg = filtered_df.groupby('Date').agg({
                    'Polarity': 'mean',
                    'Medical Score': 'mean'
                }).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_avg['Date'], y=daily_avg['Polarity'],
                                        mode='lines+markers', name='Sentiment Polarity'))
                fig.add_trace(go.Scatter(x=daily_avg['Date'], y=daily_avg['Medical Score'],
                                        mode='lines+markers', name='Medical Score'))
                fig.update_layout(title='Average Scores Over Time',
                                xaxis_title='Date',
                                yaxis_title='Score',
                                yaxis_range=[-1, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Analyses", len(filtered_df))
                with col2:
                    st.metric("Avg Polarity", f"{filtered_df['Polarity'].mean():.2f}")
                with col3:
                    st.metric("Avg Medical Score", f"{filtered_df['Medical Score'].mean():.2f}")
                with col4:
                    st.metric("Critical Cases", (filtered_df['Urgency'] == 'CRITICAL').sum())
                
                # Correlation analysis
                st.subheader("Correlation Analysis")
                # Convert urgency to numeric for correlation
                urgency_map = {'ROUTINE': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'CRITICAL': 5}
                filtered_df['Urgency_Num'] = filtered_df['Urgency'].map(urgency_map)
                
                corr = filtered_df[['Polarity', 'Medical Score', 'Word Count', 'Urgency_Num']].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                              title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Urgency breakdown
                st.subheader("Urgency Breakdown")
                urgency_summary = filtered_df.groupby('Urgency').agg({
                    'Polarity': ['mean', 'count'],
                    'Medical Score': 'mean'
                }).round(3)
                st.dataframe(urgency_summary, use_container_width=True)
            
            with tab3:
                st.subheader("Raw Data")
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download options
                csv = filtered_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="report_data.csv">📥 Download Report Data</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # Generate comprehensive report
            if st.button("📄 Generate Comprehensive Report", use_container_width=True):
                generate_comprehensive_report(filtered_df, start_date, end_date)
        else:
            st.warning("No data available for the selected date range")
    else:
        st.info("No history available to generate reports")

# Settings Page
elif page == "Settings":
    st.header("⚙️ Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analysis Settings")
        default_save = st.checkbox("Save to history by default", value=True)
        auto_wordcloud = st.checkbox("Generate word cloud automatically", value=True)
        include_vader = st.checkbox("Use VADER sentiment analysis", value=True)
        include_medical_terms = st.checkbox("Detect medical terminology", value=True)
        
        st.subheader("Display Settings")
        theme = st.selectbox("Theme", ["Light", "Dark"])
        chart_style = st.selectbox("Chart Style", ["Plotly", "Matplotlib"])
    
    with col2:
        st.subheader("Report Settings")
        include_header = st.checkbox("Include header in reports", value=True)
        include_footer = st.checkbox("Include footer in reports", value=True)
        include_medical_terms_report = st.checkbox("Include medical terms in reports", value=True)
        
        st.subheader("Data Management")
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history = []
            st.success("History cleared successfully!")
        
        if st.button("💾 Export All Data", use_container_width=True):
            # Export all data as JSON
            data = {
                'history': st.session_state.history,
                'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_analyses': len(st.session_state.history)
            }
            json_str = json.dumps(data, indent=2, default=str)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="advanced_clinical_data.json">📥 Download Advanced JSON File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("About Advanced Version")
    st.markdown("""
    **Advanced Clinical Text Analysis System v2.0**
    
    This enhanced version includes:
    
    🎯 **Multi-Method Sentiment Analysis**
    - TextBlob (general sentiment)
    - VADER (social media/clinical notes)
    - Medical-specific sentiment analysis
    
    🏥 **Clinical Features**
    - Medical terminology detection
    - Clinical urgency assessment
    - Healthcare entity extraction
    - Readability analysis for clinical documentation
    
    📊 **Enhanced Analytics**
    - Emotion intensity scoring
    - Correlation analysis
    - Trend detection
    - Comparative analysis
    
    📄 **Professional Reports**
    - Comprehensive clinical reports
    - Urgency indicators
    - Medical term highlighting
    - PDF export with all metrics
    
    For clinical research and healthcare documentation improvement.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Advanced Clinical Text Analysis System | Version 2.0 | For Healthcare Professionals and Researchers</p>
        <p>Enhanced with Multi-Method Sentiment Analysis and Clinical Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Run the app
if __name__ == "__main__":
    pass