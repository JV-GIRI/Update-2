import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import librosa.display
from io import BytesIO
import soundfile as sd
import os
import json
from datetime import datetime
import uuid
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import librosa.feature

# --- Setup ---
st.set_page_config(page_title="PCG Realtime Waveform Analyzer", layout="wide")
st.title("❤️‍🔥 GIRI'S HEARTEST - Real time PCG analyzer")

# --- Directories ---
os.makedirs("data/audio", exist_ok=True)
os.makedirs("data", exist_ok=True)
HISTORY_FILE = "data/history.json"
MODEL_PATH = "pretrained_rvhd_model.h5"  # Placeholder path to your AI model

# --- Load AI model ---
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None

# --- Helper: Save history ---
def save_history(name, age, gender, filename, timestamp, audio_buffer):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

    record_id = str(uuid.uuid4())
    audio_path = f"data/audio/{record_id}.wav"

    with open(audio_path, "wb") as f:
        f.write(audio_buffer.getbuffer())

    history.append({
        "id": record_id,
        "name": name,
        "age": age,
        "gender": gender,
        "filename": filename,
        "timestamp": timestamp,
        "audio_path": audio_path
    })

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

# --- Real-time Mic Recording ---
st.sidebar.subheader("🎙 Real-time Mic PCG Recording")
duration_sec = st.sidebar.slider("Recording duration (seconds)", 1, 10, 3)
record_button = st.sidebar.button("⏺ Record from Mic")
if record_button:
    st.sidebar.info("Recording...")
    recording = sd.rec(int(duration_sec * 44100), samplerate=44100, channels=1)
    sd.wait()
    st.sidebar.success("Recording complete.")

    mic_audio_path = f"data/audio/live_recording.wav"
    sf.write(mic_audio_path, recording, 44100)
    uploaded_file = open(mic_audio_path, "rb")

# --- Main Logic ---
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    y, sr = librosa.load(uploaded_file, sr=None)

    st.subheader("🔈 Original PCG Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Original PCG Waveform")
    st.pyplot(fig)

    st.subheader("🎚 Waveform Controls")
    duration = st.slider("Select duration (seconds)", 1, int(len(y) / sr), 5)
    amplitude_factor = st.slider("Amplitude scaling", 0.1, 5.0, 1.0)
    y_trimmed = y[:sr * duration] * amplitude_factor

    from scipy.signal import butter, filtfilt
    def bandpass_filter(data, sr, lowcut=25.0, highcut=400.0):
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, data)

    y_denoised = bandpass_filter(y_trimmed, sr)

    st.subheader("🔇 Denoised Waveform (Bandpass Filtered 25–400 Hz)")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y_denoised, sr=sr, ax=ax2, color='r')
    ax2.set(title="Filtered PCG Signal")
    st.pyplot(fig2)

    st.subheader("▶️ Play Denoised Audio")
    buf = BytesIO()
    sf.write(buf, y_denoised, sr, format='WAV')
    st.audio(buf.getvalue(), format='audio/wav')

    if submit_patient and name:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_history(name, age, gender, uploaded_file.name, timestamp, buf)
        st.sidebar.success("✅ Saved successfully!")

    # --- AI Diagnosis ---
    if model:
        st.subheader("🧠 AI Diagnosis Result")
        mfccs = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=40)
        mfccs = np.expand_dims(mfccs, axis=-1)
        mfccs = np.expand_dims(mfccs, axis=0)

        prediction = model.predict(mfccs)
        classes = ['Aortic Stenosis', 'Aortic Regurgitation', 'Mitral Stenosis', 'Mitral Regurgitation']
        pred_label = classes[np.argmax(prediction)]
        st.success(f"✅ Detected Condition: **{pred_label}**")

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
            y_old, sr_old = librosa.load(case["audio_path"], sr=None)

            st.subheader("📈 Waveform of Historical Case")
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y_old, sr=sr_old, ax=ax3, color='g')
            ax3.set(title="Historical PCG Waveform")
            st.pyplot(fig3)

            if model:
                mfccs_old = librosa.feature.mfcc(y=y_old, sr=sr_old, n_mfcc=40)
                mfccs_old = np.expand_dims(mfccs_old, axis=-1)
                mfccs_old = np.expand_dims(mfccs_old, axis=0)
                prediction_old = model.predict(mfccs_old)
                pred_label_old = classes[np.argmax(prediction_old)]
                st.success(f"🩺 AI Detected: **{pred_label_old}**")
else:
    st.sidebar.info("No past history available.")
