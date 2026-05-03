"""
Complete Local Voice Analysis App
No API keys required - runs entirely on your machine
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import queue
import threading
import time
import os
import tempfile
from datetime import datetime
import wave
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Local Voice Analysis Studio",
    page_icon="🎤",
    layout="wide"
)

# Try importing streamlit-webrtc with error handling
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError as e:
    WEBRTC_AVAILABLE = False
    st.sidebar.error(f"⚠️ WebRTC not available: {str(e)}")

# Import our custom modules
from utils.audio_processor import AudioProcessor, extract_audio_features
from utils.emotion_model import SimpleEmotionDetector as EmotionDetector
from utils.speech_to_text import LocalSTT

# Check availability
AUDIO_PROCESSOR_AVAILABLE = True
EMOTION_MODEL_AVAILABLE = True
STT_AVAILABLE = True

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'transcript_history' not in st.session_state:
    st.session_state.transcript_history = []
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = time.time()
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Custom audio processor for WebRTC
class VoiceAnalysisProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []
        self.sample_rate = 16000
        self.chunk_duration = 3  # Analyze every 3 seconds
        self.samples_per_chunk = int(self.sample_rate * self.chunk_duration)
        self.min_samples = int(self.sample_rate * 1)  # Minimum 1 second
        
    def recv(self, frame):
        """Process incoming audio frames"""
        try:
            # Convert frame to numpy array
            audio_data = frame.to_ndarray()
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            # Keep only last 5 seconds
            max_buffer = int(self.sample_rate * 5)
            if len(self.audio_buffer) > max_buffer:
                self.audio_buffer = self.audio_buffer[-max_buffer:]
            
            # If we have enough samples and not already processing
            if (len(self.audio_buffer) >= self.min_samples and 
                not st.session_state.processing and
                time.time() - st.session_state.last_analysis_time > 2):
                
                st.session_state.processing = True
                
                # Take the latest chunk
                chunk = np.array(self.audio_buffer[-self.samples_per_chunk:])
                
                # Put in queue
                st.session_state.audio_queue.put(chunk)
                
                st.session_state.last_analysis_time = time.time()
                
                # Release lock after delay
                def release_lock():
                    time.sleep(1)
                    st.session_state.processing = False
                
                threading.Thread(target=release_lock, daemon=True).start()
                
        except Exception as e:
            print(f"Error in audio processor: {e}")
        
        return frame

@st.cache_resource
def load_emotion_model():
    """Load emotion detection model"""
    if EMOTION_MODEL_AVAILABLE:
        try:
            return EmotionDetector()
        except Exception as e:
            st.error(f"Failed to load emotion model: {e}")
            return None
    return None

@st.cache_resource
def load_stt_model():
    """Load speech-to-text model"""
    if STT_AVAILABLE:
        try:
            return LocalSTT()
        except Exception as e:
            st.error(f"Failed to load STT model: {e}")
            return None
    return None

def main():
    # Sidebar
    with st.sidebar:
        st.title("🎤 Voice Analysis Studio")
        st.markdown("---")
        
        # Model status
        st.subheader("📊 Model Status")
        
        # Load models
        if not st.session_state.models_loaded:
            with st.spinner("Loading models..."):
                emotion_model = load_emotion_model()
                stt_model = load_stt_model()
                
                st.session_state.emotion_model = emotion_model
                st.session_state.stt_model = stt_model
                st.session_state.models_loaded = True
        else:
            emotion_model = st.session_state.emotion_model
            stt_model = st.session_state.stt_model
        
        # Display model status
        col1, col2 = st.columns(2)
        with col1:
            if emotion_model is not None:
                st.success("✅ Emotion")
            else:
                st.error("❌ Emotion")
        
        with col2:
            if stt_model is not None:
                st.success("✅ STT")
            else:
                st.error("❌ STT")
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        analysis_mode = st.multiselect(
            "Analysis Features",
            ["Emotion Detection", "Speech-to-Text"],
            default=["Emotion Detection", "Speech-to-Text"]
        )
        
        st.markdown("---")
        
        # Real-time settings
        st.subheader("🎛️ Real-time Settings")
        analysis_interval = st.slider(
            "Analysis Interval (seconds)",
            min_value=2,
            max_value=5,
            value=3,
            step=1
        )
        
        st.markdown("---")
        
        # WebRTC Status
        if not WEBRTC_AVAILABLE:
            st.warning("⚠️ Real-time capture unavailable. Use file upload instead.")
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown(
                """
                **Local Voice Analysis Studio**
                
                This app runs completely offline using:
                - Rule-based emotion detection
                - Whisper for speech-to-text
                
                **No data leaves your computer!**
                """
            )
    
    # Main content area
    st.title("🎤 Real-Time Voice Analysis")
    st.markdown("Analyze voice in real-time using local AI models")
    
    # Check if any models are available
    if not any([emotion_model, stt_model]):
        st.error("❌ No models could be loaded. Please check your installation and try again.")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🎙️ Live Analysis", 
        "📊 Analytics", 
        "📜 History"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎙️ Live Voice Input")
            
            # WebRTC streamer for real-time audio
            if WEBRTC_AVAILABLE:
                # Create processor
                processor = VoiceAnalysisProcessor()
                st.session_state.processor = processor
                
                # WebRTC streamer - FIXED: removed ctx.stop()
                ctx = webrtc_streamer(
                    key="voice-analysis",
                    mode=WebRtcMode.SENDONLY,
                    audio_processor_factory=lambda: processor,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    },
                    media_stream_constraints={
                        "audio": {
                            "echoCancellation": True,
                            "noiseSuppression": True,
                            "autoGainControl": True
                        },
                        "video": False
                    },
                    async_processing=True
                )
                
                # Show recording status - FIXED: removed ctx.stop()
                if ctx and ctx.state.playing:
                    st.success("🔴 Recording in progress... Speak now!")
                    st.info("⏹️ To stop, click the 'STOP' button in the WebRTC controls above")
                else:
                    st.info("⏸️ Click 'START' in the WebRTC controls above to begin recording")
            else:
                st.info("ℹ️ Real-time capture not available. Please use file upload below.")
            
            # Manual file upload option
            st.markdown("---")
            st.subheader("📁 Or upload audio file")
            uploaded_file = st.file_uploader(
                "Choose audio file", 
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac']
            )
        
        with col2:
            st.subheader("📊 Current Analysis")
            
            # Create placeholders
            status_placeholder = st.empty()
            emotion_placeholder = st.empty()
            transcript_placeholder = st.empty()
            
            # Process real-time audio from queue
            if WEBRTC_AVAILABLE and ctx and ctx.state.playing:
                try:
                    # Check if there's audio in the queue
                    if not st.session_state.audio_queue.empty():
                        audio_chunk = st.session_state.audio_queue.get()
                        
                        status_placeholder.info("🔄 Analyzing...")
                        
                        results = {}
                        
                        # Emotion Detection
                        if "Emotion Detection" in analysis_mode and emotion_model is not None:
                            emotion, confidence = emotion_model.predict(audio_chunk, 16000)
                            results['emotion'] = (emotion, confidence)
                            st.session_state.emotion_history.append({
                                'timestamp': datetime.now(),
                                'emotion': emotion,
                                'confidence': confidence
                            })
                        
                        # Speech-to-Text
                        if "Speech-to-Text" in analysis_mode and stt_model is not None:
                            text = stt_model.transcribe(audio_chunk, 16000)
                            results['text'] = text
                            if text and text not in ["", "[Whisper not installed]", "[No speech detected]"]:
                                st.session_state.transcript_history.append({
                                    'timestamp': datetime.now(),
                                    'text': text
                                })
                        
                        # Display results
                        if 'emotion' in results:
                            emotion, conf = results['emotion']
                            emotion_placeholder.metric(
                                "Detected Emotion", 
                                f"{emotion.title()} ({conf:.1%})"
                            )
                            
                            # Emotion color
                            colors = {
                                'happy': '🟢', 'sad': '🔵', 'angry': '🔴',
                                'fear': '🟤', 'disgust': '🟣', 'surprise': '🟡',
                                'neutral': '⚪', 'calm': '🔵'
                            }
                            emotion_placeholder.markdown(
                                f"{colors.get(emotion.lower(), '⚪')} **{emotion.title()}**"
                            )
                        
                        if 'text' in results:
                            transcript_placeholder.markdown("**Transcript:**")
                            if results['text'] and results['text'] not in ["", "[Whisper not installed]", "[No speech detected]"]:
                                transcript_placeholder.info(results['text'])
                            else:
                                transcript_placeholder.info("🎤 Listening...")
                        
                        status_placeholder.success("✅ Analysis complete!")
                        
                    else:
                        # Show waiting message
                        transcript_placeholder.info("🎤 Listening for speech...")
                except Exception as e:
                    status_placeholder.error(f"Error: {str(e)}")
            
            # Process uploaded file
            elif uploaded_file is not None:
                try:
                    import soundfile as sf
                    audio_bytes = uploaded_file.read()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_path = tmp_file.name
                    
                    audio_array, sample_rate = sf.read(tmp_path)
                    os.unlink(tmp_path)
                    
                    status_placeholder.info("🔄 Analyzing...")
                    
                    results = {}
                    
                    if "Emotion Detection" in analysis_mode and emotion_model is not None:
                        emotion, confidence = emotion_model.predict(audio_array, sample_rate)
                        results['emotion'] = (emotion, confidence)
                        st.session_state.emotion_history.append({
                            'timestamp': datetime.now(),
                            'emotion': emotion,
                            'confidence': confidence
                        })
                    
                    if "Speech-to-Text" in analysis_mode and stt_model is not None:
                        text = stt_model.transcribe(audio_array, sample_rate)
                        results['text'] = text
                        st.session_state.transcript_history.append({
                            'timestamp': datetime.now(),
                            'text': text
                        })
                    
                    status_placeholder.success("✅ Analysis complete!")
                    
                    if 'emotion' in results:
                        emotion, conf = results['emotion']
                        emotion_placeholder.metric(
                            "Detected Emotion", 
                            f"{emotion.title()} ({conf:.1%})"
                        )
                        
                        colors = {
                            'happy': '🟢', 'sad': '🔵', 'angry': '🔴',
                            'fear': '🟤', 'disgust': '🟣', 'surprise': '🟡',
                            'neutral': '⚪', 'calm': '🔵'
                        }
                        emotion_placeholder.markdown(
                            f"{colors.get(emotion.lower(), '⚪')} **{emotion.title()}**"
                        )
                    
                    if 'text' in results:
                        transcript_placeholder.markdown("**Transcript:**")
                        transcript_placeholder.info(results['text'] if results['text'] else "[No speech detected]")
                    
                except Exception as e:
                    status_placeholder.error(f"Error processing file: {e}")
    
    with tab2:
        st.subheader("📊 Voice Analysis Analytics")
        
        if st.session_state.emotion_history:
            df_emotions = pd.DataFrame(st.session_state.emotion_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                emotion_counts = df_emotions['emotion'].value_counts()
                fig_pie = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Emotion Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_line = px.line(
                    df_emotions,
                    x='timestamp',
                    y='confidence',
                    color='emotion',
                    title="Emotion Confidence Over Time"
                )
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No emotion data yet. Start recording to see analytics!")
        
        if st.session_state.transcript_history:
            st.subheader("📝 Recent Transcripts")
            df_transcripts = pd.DataFrame(st.session_state.transcript_history[-5:])
            for _, row in df_transcripts.iterrows():
                st.text(f"{row['timestamp'].strftime('%H:%M:%S')}: {row['text']}")
    
    with tab3:
        st.subheader("📜 Analysis History")
        
        if st.session_state.emotion_history:
            with st.expander("📊 Emotion History", expanded=True):
                df_emotions = pd.DataFrame(st.session_state.emotion_history)
                st.dataframe(
                    df_emotions.tail(10).style.format({'confidence': '{:.1%}'}),
                    use_container_width=True
                )
        
        if st.session_state.transcript_history:
            with st.expander("📝 Transcript History", expanded=True):
                df_transcripts = pd.DataFrame(st.session_state.transcript_history)
                st.dataframe(df_transcripts.tail(10), use_container_width=True)
        
        if st.button("🗑️ Clear All History", type="primary"):
            st.session_state.emotion_history = []
            st.session_state.transcript_history = []
            st.success("History cleared!")
            st.rerun()

if __name__ == "__main__":
    main()