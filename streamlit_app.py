import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import librosa.display
from io import BytesIO
import soundfile as sf
import os
import json
from datetime import datetime

st.set_page_config(page_title="PCG Realtime Waveform Analyzer", layout="wide")
st.title("🔬 Real-time PCG Waveform & Noise Reduction")

# Folder to save patient cases
HISTORY_FOLDER = "patient_history"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# --- Sidebar for Patient Information ---
with st.sidebar:
    st.header("👤 Add Patient Information")
    patient_name = st.text_input("Name")
    patient_age = st.number_input("Age", min_value=0, step=1)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    record_button = st.button("💾 Save Patient Details")

    if record_button:
        if not patient_name:
            st.warning("Enter patient name.")
        else:
            patient_id = f"{patient_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state['patient_id'] = patient_id

            data = {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "timestamp": datetime.now().isoformat()
            }
            with open(f"{HISTORY_FOLDER}/{patient_id}.json", "w") as f:
                json.dump(data, f)
            st.success(f"✅ Saved: {patient_name}")

# --- Upload or Record PCG Audio ---
st.subheader("📤 Upload or Record PCG (.wav)")
uploaded_file = st.file_uploader("Choose a PCG file", type=["wav"])

# --- PCG Audio Analysis Workflow ---
def analyze_audio(file_data, sr=None):
    # Load and show original waveform
    y, sr = librosa.load(file_data, sr=sr)

    st.subheader("🔈 Original PCG Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Original PCG Waveform")
    st.pyplot(fig)

    # Controls
    st.subheader("🎚 Waveform Controls")
    duration = st.slider("Select duration (seconds)", 1, int(len(y)/sr), 5)
    amplitude_factor = st.slider("Amplitude scaling", 0.1, 5.0, 1.0)

    y_trimmed = y[:sr * duration] * amplitude_factor

    # Bandpass filter
    from scipy.signal import butter, filtfilt
    def bandpass_filter(data, sr, lowcut=25.0, highcut=400.0):
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, data)

    y_denoised = bandpass_filter(y_trimmed, sr)

    st.subheader("🔇 Denoised Waveform (25–400 Hz Bandpass)")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y_denoised, sr=sr, ax=ax2, color='r')
    ax2.set(title="Filtered PCG Signal")
    st.pyplot(fig2)

    st.subheader("▶️ Play Denoised Audio")
    buf = BytesIO()
    sf.write(buf, y_denoised, sr, format='WAV')
    st.audio(buf.getvalue(), format='audio/wav')

    # Save waveform with patient info if available
    if 'patient_id' in st.session_state:
        audio_path = f"{HISTORY_FOLDER}/{st.session_state['patient_id']}.wav"
        with open(audio_path, "wb") as f:
            f.write(buf.getvalue())

# --- If file uploaded ---
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    analyze_audio(uploaded_file)

# --- History Viewer ---
st.subheader("📁 View Patient History")
history_files = [f for f in os.listdir(HISTORY_FOLDER) if f.endswith(".json")]

if history_files:
    selected_case = st.selectbox("Select saved patient", history_files)
    if selected_case:
        with open(f"{HISTORY_FOLDER}/{selected_case}") as f:
            data = json.load(f)
        st.write("👤 Patient Info")
        st.write(f"**Name:** {data['name']}")
        st.write(f"**Age:** {data['age']}")
        st.write(f"**Gender:** {data['gender']}")
        st.write(f"**Timestamp:** {data['timestamp']}")

        # Load associated audio
        audio_file = selected_case.replace(".json", ".wav")
        audio_path = os.path.join(HISTORY_FOLDER, audio_file)
        if os.path.exists(audio_path):
            st.audio(audio_path, format='audio/wav')
            analyze_audio(audio_path)
        else:
            st.warning("No audio file found for this patient.")
else:
    st.info("No history yet. Upload and save a new case.")
