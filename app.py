"""Streamlit UI for voice recognition POC - Fixed Version."""

import streamlit as st
import numpy as np
from pathlib import Path
import config
from voice_service import VoiceService
from datetime import datetime
import soundfile as sf
import tempfile
import time


# Page config
st.set_page_config(
    page_title="Voice Recognition System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize service
@st.cache_resource
def get_service():
    return VoiceService()

service = get_service()

# Initialize session state
if 'current_enrollment_audio' not in st.session_state:
    st.session_state.current_enrollment_audio = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'processed_audio' not in st.session_state:
    st.session_state.processed_audio = None


def identify_speaker_file(audio_file):
    """Identify speaker from uploaded audio file."""
    try:
        # Read audio file
        audio_array, sr = sf.read(audio_file)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Check minimum length (need at least 1 second)
        min_samples = sr * 1
        if len(audio_array) < min_samples:
            st.error(f"⚠️ Audio too short! Need at least 1 second. Got {len(audio_array)/sr:.1f}s")
            return None, None, None, None
        
        # Use longer segment for better accuracy (up to 10 seconds)
        max_samples = sr * 10
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        
        # Identify
        is_known, client_id, name, confidence = service.identify_from_array(audio_array, sr)
        
        # Store for enrollment if unknown
        if not is_known:
            st.session_state.current_enrollment_audio = (audio_array, sr)
        
        return is_known, client_id, name, confidence
        
    except Exception as e:
        st.error(f"⚠️ Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def enroll_speaker(name):
    """Enroll new speaker."""
    if not name or not name.strip():
        return False, "Please enter a name!"
    
    if st.session_state.current_enrollment_audio is None:
        return False, "No audio recorded. Please record your voice first!"
    
    audio_array, sr = st.session_state.current_enrollment_audio
    
    try:
        client_id = service.enroll_from_array(audio_array, name.strip(), sr)
        st.session_state.current_enrollment_audio = None
        return True, f"✅ Successfully enrolled as {name}! (ID: {client_id})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"❌ Enrollment failed: {str(e)}"


def delete_speaker(client_id):
    """Delete a speaker from the system."""
    try:
        service.mongo.users.delete_one({"client_id": client_id})
        point_id = hash(client_id) & 0x7FFFFFFF
        service.qdrant.client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=[point_id]
        )
        return True, "✅ Speaker deleted successfully!"
    except Exception as e:
        return False, f"❌ Failed to delete: {str(e)}"


def get_sample_speakers():
    """Get list of ALL sample speakers from database."""
    try:
        all_speakers = service.get_all_speakers()
        
        # Get samples by BOTH is_sample flag AND client_id pattern
        sample_speakers = [
            s for s in all_speakers 
            if s.get('is_sample', False) or s.get('client_id', '').startswith('SAMPLE_')
        ]
        
        speakers = []
        for speaker in sample_speakers:
            name = speaker.get('name', 'Unknown')
            client_id = speaker.get('client_id', '')
            
            # Try to find corresponding audio file
            try:
                parts = client_id.split('_')
                if len(parts) >= 2 and parts[0] == 'SAMPLE':
                    sample_num = parts[1]
                    audio_path = config.SAMPLES_DIR / f"sample_{sample_num}.wav"
                    
                    if audio_path.exists():
                        speakers.append((name, str(audio_path), client_id))
                    else:
                        speakers.append((name, None, client_id))
                else:
                    # Non-standard naming, still include
                    speakers.append((name, None, client_id))
            except:
                speakers.append((name, None, client_id))
        
        # Sort by client_id for consistent ordering
        speakers.sort(key=lambda x: x[2])
        
        return speakers
    except Exception as e:
        print(f"Error loading speakers: {e}")
        import traceback
        traceback.print_exc()
        return []


# Header
st.markdown('<p class="main-header">🎙️ Voice Recognition System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Multi-Speaker Identification & Auto-Enrollment</p>', unsafe_allow_html=True)

# Show current threshold setting
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown(f"**Current Similarity Threshold:** {config.SIMILARITY_THRESHOLD}")
    
    new_threshold = st.slider(
        "Adjust Recognition Threshold",
        min_value=0.50,
        max_value=0.95,
        value=config.SIMILARITY_THRESHOLD,
        step=0.05,
        help="Higher = stricter matching (fewer false positives, more false negatives)"
    )
    
    if new_threshold != config.SIMILARITY_THRESHOLD:
        config.SIMILARITY_THRESHOLD = new_threshold
        st.success(f"✅ Threshold updated to {new_threshold}")
    
    st.markdown("---")
    st.markdown("**Recommended Values:**")
    st.caption("• 0.70-0.75: Balanced (default)")
    st.caption("• 0.80-0.85: Strict (better accuracy)")
    st.caption("• 0.60-0.65: Lenient (more flexible)")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Live Recognition",
    "🧪 Test Samples",
    "👥 Speaker Database"
])

# ==================== TAB 1: Real-Time Identification ====================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 🎤 Voice Recording")
        st.caption("⏱️ Record 3-5 seconds of clear speech for best results")
        
        # Use native audio input
        audio_file = st.file_uploader(
            "Upload audio or use microphone below",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            key="audio_upload",
            help="Upload a voice recording"
        )
        
        st.markdown("**OR**")
        
        # Native audio recording
        recorded_audio = st.audio_input("Record your voice", key="audio_recorder")
        
        # Process whichever is available
        audio_to_process = recorded_audio if recorded_audio else audio_file
        
        if audio_to_process:
            st.audio(audio_to_process)
            
            if st.button("🔍 Identify Speaker", key="identify_btn", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing voice pattern..."):
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(audio_to_process.read())
                        tmp_path = tmp.name
                    
                    # Process
                    is_known, client_id, name, confidence = identify_speaker_file(tmp_path)
                    
                    if is_known is not None:
                        st.session_state.last_result = (is_known, client_id, name, confidence)
                        st.rerun()
        else:
            st.info("👆 Click 'Record your voice' or upload an audio file")
    
    with col2:
        st.markdown("### 📊 Recognition Results")
        
        if st.session_state.last_result:
            is_known, client_id, name, confidence = st.session_state.last_result
            
            if is_known:
                # KNOWN SPEAKER
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"### ✅ Speaker Identified")
                st.markdown(f"**Name:** {name}")
                st.markdown(f"**Client ID:** `{client_id}`")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show confidence with color coding
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col_b:
                    if confidence >= 0.85:
                        st.success("High Confidence ✅")
                    elif confidence >= 0.70:
                        st.info("Good Match ✓")
                    else:
                        st.warning("Low Confidence ⚠️")
                
                st.progress(confidence)
                
                if confidence < 0.80:
                    st.caption("💡 Tip: Lower confidence may indicate similar voices. Consider re-recording with more speech.")
                
            else:
                # UNKNOWN SPEAKER
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("### 🆕 New Speaker Detected!")
                st.markdown("This voice is not in our database.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("#### 📝 Enroll New Speaker")
                
                name_input = st.text_input(
                    "Full Name",
                    placeholder="e.g., Sarah Johnson",
                    key="name_input_auto"
                )
                
                if st.button("✅ Enroll Me", key="enroll_btn_auto", type="primary", use_container_width=True):
                    if name_input:
                        success, message = enroll_speaker(name_input)
                        if success:
                            st.success(message)
                            st.session_state.last_result = None
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("⚠️ Please enter your name!")
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Waiting for voice input...**")
            st.markdown("Upload or record audio, then click 'Identify Speaker'")
            st.markdown('</div>', unsafe_allow_html=True)


# ==================== TAB 2: Test with Samples ====================
with tab2:
    st.markdown("### 🧪 Test Pre-Enrolled Voice Samples")
    
    sample_speakers = get_sample_speakers()
    
    # Show database statistics
    all_speakers = service.get_all_speakers()
    total_samples = len([
        s for s in all_speakers 
        if s.get('is_sample', False) or s.get('client_id', '').startswith('SAMPLE_')
    ])
    samples_with_audio = len([s for s in sample_speakers if s[1] is not None])
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Enrolled Samples", total_samples)
    with col_stat2:
        st.metric("Samples with Audio Files", samples_with_audio)
    with col_stat3:
        st.metric("Ready to Test", samples_with_audio)
    
    st.markdown("---")
    
    if sample_speakers:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### 🎵 Sample Audio Library")
            st.caption(f"Showing all {len(sample_speakers)} enrolled samples")
            
            # Create display options
            speaker_display = []
            for idx, (name, audio_path, client_id) in enumerate(sample_speakers):
                status = "✅" if audio_path else "⚠️ (no audio)"
                speaker_display.append(f"{status} {name}")
            
            selected_index = st.selectbox(
                "Select a speaker to test",
                range(len(speaker_display)),
                format_func=lambda i: speaker_display[i],
                key="sample_select"
            )
            
            selected_name, selected_path, selected_client_id = sample_speakers[selected_index]
            
            # Show client ID
            st.caption(f"Client ID: `{selected_client_id}`")
            
            if selected_path and Path(selected_path).exists():
                st.audio(selected_path)
                
                if st.button("🧪 Run Test", key="test_btn", type="primary", use_container_width=True):
                    with st.spinner("🔄 Testing..."):
                        try:
                            is_known, client_id, name, confidence = service.identify_speaker(selected_path)
                            st.session_state.test_result = (is_known, client_id, name, confidence, selected_name, selected_client_id)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Test failed: {str(e)}")
                            import traceback
                            traceback.print_exc()
            else:
                st.warning(f"⚠️ Audio file not found for {selected_name}")
                st.info("This speaker is enrolled in the database but the audio file is missing.")
        
        with col2:
            st.markdown("#### 📊 Test Results")
            
            if 'test_result' in st.session_state and st.session_state.test_result:
                is_known, client_id, name, confidence, expected_name, expected_id = st.session_state.test_result
                
                if is_known:
                    match = (client_id == expected_id)
                    
                    if match:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"### ✅ Correct Match!")
                        st.markdown(f"**Identified:** {name}")
                        st.markdown(f"**Client ID:** `{client_id}`")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown(f"### ⚠️ Mismatch!")
                        st.markdown(f"**Identified:** {name} (`{client_id}`)")
                        st.markdown(f"**Expected:** {expected_name} (`{expected_id}`)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col_b:
                        if confidence >= 0.85:
                            st.success("High Confidence ✅")
                        elif confidence >= 0.70:
                            st.info("Good Match ✓")
                        else:
                            st.warning("Low Confidence ⚠️")
                    
                    st.progress(confidence)
                    
                    if not match:
                        st.caption("💡 Tip: Mismatches can occur with similar voices. Try adjusting the threshold or re-enrolling with better quality samples.")
                else:
                    st.error(f"❌ Not Recognized")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    st.markdown(f"**Expected:** {expected_name}")
                    st.caption(f"Below threshold ({config.SIMILARITY_THRESHOLD}). Try lowering the threshold in sidebar.")
            else:
                st.info("Select a speaker and click 'Run Test'")
    else:
        st.warning("⚠️ No sample speakers found in database!")
        st.info("Run `python data_setup.py` to enroll sample speakers")


# ==================== TAB 3: Database ====================
with tab3:
    st.markdown("### 👥 Speaker Database")
    
    if st.button("🔄 Refresh", key="refresh_btn"):
        st.rerun()
    
    speakers = service.get_all_speakers()
    
    if speakers:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total", len(speakers))
        with col2:
            samples = len([s for s in speakers if s.get('client_id', '').startswith('SAMPLE_') or s.get('is_sample', False)])
            st.metric("Samples", samples)
        with col3:
            st.metric("Users", len(speakers) - samples)
        
        st.markdown("---")
        
        for idx, speaker in enumerate(speakers, 1):
            name = speaker.get('name', 'N/A')
            client_id = speaker.get('client_id', 'N/A')
            created = str(speaker.get('created_at', 'N/A'))[:19]
            is_sample = client_id.startswith('SAMPLE_') or speaker.get('is_sample', False)
            
            col1, col2, col3, col4 = st.columns([0.5, 2, 2.5, 1])
            
            with col1:
                st.write(f"**{idx}**")
            with col2:
                badge = "🎵" if is_sample else "👤"
                st.write(f"{badge} **{name}**")
            with col3:
                st.caption(f"`{client_id}` | {created}")
            with col4:
                if st.button("🗑️", key=f"del_{client_id}", use_container_width=True):
                    success, msg = delete_speaker(client_id)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            
            st.divider()
    else:
        st.info("🔭 No speakers enrolled yet")


# Footer
st.markdown("---")
st.caption("Voice Recognition System v1.0 | SpeechBrain • Qdrant • MongoDB")