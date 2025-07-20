import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa
import os
import io
import json
from datetime import datetime

st.set_page_config(page_title="AI-based RVHD Detection", layout="wide")

# -- Load / Save History from a JSON file
HISTORY_FILE = "patient_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# -- Title
st.title("🫀 AI-based Rheumatic Valvular Heart Disease Detection")

# -- Sidebar for Patient Info
with st.sidebar:
    st.header("🧍 Patient Information")
    patient_name = st.text_input("Name")
    patient_age = st.number_input("Age", min_value=0, max_value=120, step=1)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    patient_notes = st.text_area("Clinical Notes")
    save_info = st.button("💾 Save Patient Info")

# -- Upload PCG File
uploaded_file = st.file_uploader("📂 Upload Heart Sound (PCG) File", type=["wav", "mp3", "ogg"])

# -- Infrasonic Recorder Info
st.markdown("🎙️ Want to use REDVOX app for infrasonic recording? [Learn more](https://www.redvox.io/)")

# -- History View
st.sidebar.header("📜 Patient History")
if "history" not in st.session_state:
    st.session_state["history"] = load_history()

selected_case = st.sidebar.selectbox("Select previous case", [""] + [f"{i+1}. {case['name']} ({case['timestamp']})" 
                                                for i, case in enumerate(st.session_state["history"])])

if selected_case and selected_case != "":
    index = int(selected_case.split('.')[0]) - 1
    selected = st.session_state["history"][index]
    st.subheader("📝 Previous Patient Details")
    st.text(f"Name: {selected['name']}")
    st.text(f"Age: {selected['age']}")
    st.text(f"Gender: {selected['gender']}")
    st.text(f"Notes: {selected['notes']}")
    st.audio(io.BytesIO(bytes.fromhex(selected['audio_data'])), format='audio/wav')

# -- Save Patient Info + PCG
if uploaded_file and save_info:
    audio_data = uploaded_file.read()
    case = {
        "name": patient_name,
        "age": patient_age,
        "gender": patient_gender,
        "notes": patient_notes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio_data": audio_data.hex()
    }
    st.session_state["history"].append(case)
    save_history(st.session_state["history"])
    st.success("Patient info & PCG saved to history ✅")

# -- PCG Analysis
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    y, sr = librosa.load(uploaded_file, sr=None)
    duration = len(y) / sr

    st.subheader("📉 PCG Waveform")
    # Adjustment controls
    start_sec = st.slider("Start Time (sec)", 0.0, float(duration), 0.0, step=0.1)
    end_sec = st.slider("End Time (sec)", 0.0, float(duration), float(duration), step=0.1)
    amplitude_gain = st.slider("Amplitude Gain", 0.1, 5.0, 1.0, step=0.1)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    y_display = y[start_sample:end_sample] * amplitude_gain
    t_display = np.linspace(start_sec, end_sec, len(y_display))

    fig, ax = plt.subplots()
    ax.plot(t_display, y_display)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Heart Sound Waveform")
    st.pyplot(fig)

    # -- Noise Reduction
    st.subheader("🧹 Noise Reduction")
    if st.button("Reduce Noise (Bandpass Filter)"):
        sos = signal.butter(10, [20, 500], btype='bandpass', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, y)
        st.audio(librosa.util.buf_to_float(filtered), format='audio/wav')
        st.success("Noise reduced successfully!")
