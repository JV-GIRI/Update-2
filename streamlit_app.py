import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.signal import butter, lfilter
from datetime import datetime
import json
import tempfile

st.set_page_config(layout="wide")
st.title("🩺 PCG Analyzer with History + Patient Info")

# Directories
UPLOAD_FOLDER = "uploaded_audios"
PATIENT_DATA = "patient_data.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load/save patient data
def save_patient_data(data):
    if os.path.exists(PATIENT_DATA):
        with open(PATIENT_DATA, "r") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(data)
    with open(PATIENT_DATA, "w") as f:
        json.dump(existing, f)

def load_patient_data():
    if os.path.exists(PATIENT_DATA):
        with open(PATIENT_DATA, "r") as f:
            return json.load(f)
    return []

# Noise reduction
def reduce_noise(audio, sr):
    b, a = butter(6, 0.05)
    return lfilter(b, a, audio)

# Save modified audio to temp file for playback
def save_temp_audio(audio_data, sr):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, sr)
    return temp_file.name

# Audio analysis
def analyze_audio(path, unique_id):
    sr, audio = wav.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    st.subheader("🔊 Audio Waveform")

    col1, col2 = st.columns(2)
    with col1:
        amplitude_factor = st.slider("Amplitude scaling", 0.1, 5.0, 1.0, key=f"amp_{unique_id}")
    with col2:
        duration_slider = st.slider("Adjust duration (seconds)", 1, int(len(audio) / sr), 5, key=f"dur_{unique_id}")

    adjusted_audio = audio[:duration_slider * sr] * amplitude_factor

    # Reduce noise
    if st.button("🧹 Reduce Noise", key=f"noise_{unique_id}"):
        adjusted_audio = reduce_noise(adjusted_audio, sr)
        st.info("Noise reduction applied.")

    # Amplify audio
    if st.button("🔊 Amplify Audio", key=f"amplify_{unique_id}"):
        adjusted_audio *= 2.5  # Amplification factor
        st.info("Audio amplified for better listening.")

    # Plot waveform
    fig, ax = plt.subplots()
    times = np.linspace(0, duration_slider, len(adjusted_audio))
    ax.plot(times, adjusted_audio)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Save and play modified audio
    temp_path = save_temp_audio(adjusted_audio, sr)
    st.audio(temp_path, format="audio/wav")

# Upload sidebar
st.sidebar.header("📁 Upload or Record")
upload_file = st.sidebar.file_uploader("Upload WAV File", type=["wav"])

if upload_file:
    path = os.path.join(UPLOAD_FOLDER, upload_file.name)
    with open(path, "wb") as f:
        f.write(upload_file.getbuffer())
    st.success("File uploaded successfully!")
    st.audio(path, format="audio/wav")  # Preview before analysis

# Patient Info
if "patient_saved" not in st.session_state:
    st.session_state["patient_saved"] = False

with st.sidebar.expander("🧾 Add Patient Info"):
    name = st.text_input("Name")
    age = st.number_input("Age", 1, 120)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    notes = st.text_area("Clinical Notes")

    if st.button("💾 Save Patient Case"):
        if upload_file:
            data = {
                "name": name,
                "age": age,
                "gender": gender,
                "notes": notes,
                "file": upload_file.name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_patient_data(data)
            st.session_state["patient_saved"] = True
            st.success("Patient data saved. Now analyzing the audio...")
        else:
            st.warning("Please upload a PCG file before saving.")

# Perform analysis only after case is saved
if upload_file and st.session_state["patient_saved"]:
    analyze_audio(path, unique_id=upload_file.name)

# History
st.subheader("📚 Case History")
patient_data = load_patient_data()
if patient_data:
    for i, entry in enumerate(patient_data[::-1]):
        with st.expander(f"{entry['name']} ({entry['age']} y/o) - {entry['date']}"):
            st.write(f"**Gender:** {entry['gender']}")
            st.write(f"**Notes:** {entry['notes']}")
            file_path = os.path.join(UPLOAD_FOLDER, entry["file"])
            if os.path.exists(file_path):
                st.audio(file_path, format="audio/wav")
                analyze_audio(file_path, unique_id=f"history_{i}")
            else:
                st.error("Audio file missing.")
else:
    st.info("No history records found.")
