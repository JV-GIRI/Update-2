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
import uuid

# --- Setup ---
st.set_page_config(page_title="PCG Realtime Waveform Analyzer", layout="wide")
st.title("🔬 Real-time PCG Waveform & Noise Reduction")

# --- Directories ---
os.makedirs("data/audio", exist_ok=True)
os.makedirs("data", exist_ok=True)
HISTORY_FILE = "data/history.json"

# --- Helper: Save history ---
def save_history(name, age, gender, filename, timestamp, audio_buffer):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

    # Generate unique ID for each patient record
    record_id = str(uuid.uuid4())
    audio_path = f"data/audio/{record_id}.wav"

    # Save audio to file
    with open(audio_path, "wb") as f:
        f.write(audio_buffer.getbuffer())

    # Append new record
    history.append({
        "id": record_id,
        "name": name,
        "age": age,
        "gender": gender,
        "filename": filename,
        "timestamp": timestamp,
        "audio_path": audio_path
    })

    # Save JSON
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# --- Sidebar Patient Info ---
st.sidebar.title("🧑‍⚕️ Patient Information")
with st.sidebar.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    submit_patient = st.form_submit_button("💾 Save Patient + PCG")

# --- Upload PCG ---
uploaded_file = st.file_uploader("📤 Upload a PCG (.wav) file", type=["wav"])

# --- Main Logic ---
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    y, sr = librosa.load(uploaded_file, sr=None)

    # Show original waveform
    st.subheader("🔈 Original PCG Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Original PCG Waveform")
    st.pyplot(fig)

    # --- Controls ---
    st.subheader("🎚 Waveform Controls")
    duration = st.slider("Select duration (seconds)", 1, int(len(y) / sr), 5)
    amplitude_factor = st.slider("Amplitude scaling", 0.1, 5.0, 1.0)

    y_trimmed = y[:sr * duration] * amplitude_factor

    # --- Denoising ---
    from scipy.signal import butter, filtfilt

    def bandpass_filter(data, sr, lowcut=25.0, highcut=400.0):
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, data)

    y_denoised = bandpass_filter(y_trimmed, sr)

    # --- Denoised Waveform ---
    st.subheader("🔇 Denoised Waveform (Bandpass Filtered 25–400 Hz)")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y_denoised, sr=sr, ax=ax2, color='r')
    ax2.set(title="Filtered PCG Signal")
    st.pyplot(fig2)

    # --- Denoised Audio Output ---
    st.subheader("▶️ Play Denoised Audio")
    buf = BytesIO()
    sf.write(buf, y_denoised, sr, format='WAV')
    st.audio(buf.getvalue(), format='audio/wav')

    # Save if form was submitted
    if submit_patient and name:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_history(name, age, gender, uploaded_file.name, timestamp, buf)
        st.sidebar.success("✅ Saved successfully!")

# --- HISTORY SECTION ---
st.sidebar.title("📂 Past Cases")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history_data = json.load(f)

    if history_data:
        case = st.sidebar.selectbox("Select a case to re-analyze", history_data[::-1], format_func=lambda x: f"{x['name']} ({x['timestamp']})")
        if case:
            st.markdown("## 📝 Previously Saved Case")
            st.markdown(f"**Name:** {case['name']}  \n**Age:** {case['age']}  \n**Gender:** {case['gender']}  \n**Timestamp:** {case['timestamp']}")

            st.audio(case["audio_path"], format='audio/wav')

else:
    st.sidebar.info("No past history available.")
