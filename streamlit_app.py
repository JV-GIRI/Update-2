import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
from io import BytesIO
import datetime
import uuid

# Storage for patient data and cases
if 'cases' not in st.session_state:
    st.session_state.cases = []

# ------------------ Patient Information Form ------------------
st.set_page_config(page_title="RVHD PCG Analyzer", layout="wide")
st.title("💓 AI-Based RVHD Detection from PCG Recordings")

st.sidebar.title("📋 Patient Information")
with st.sidebar.form(key="patient_form"):
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    date = st.date_input("Examination Date", value=datetime.date.today())
    submit_info = st.form_submit_button(label="Start PCG Analysis")

# ------------------ File Upload and Recorder ------------------
st.header("🩺 Upload or Record Heart Sound")
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload PCG WAV File", type=[".wav"])

with col2:
    st.write("📡 Infrasonic Recorder")
    st.info("(Feature Placeholder – Add custom infrasonic recording module here)")

# ------------------ Audio Processing Functions ------------------
def denoise_audio(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    threshold_hz = 250
    mask = cent.mean() > threshold_hz
    return y * mask

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def display_waveform(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(f"Waveform ({duration:.2f} seconds)")
    st.pyplot(fig)

# ------------------ Simple AI Analysis Logic ------------------
def dummy_rvhd_model(feature_vector):
    prob = np.mean(feature_vector) % 1
    if prob > 0.6:
        return "Mitral Stenosis"
    elif prob > 0.4:
        return "Aortic Regurgitation"
    elif prob > 0.2:
        return "Aortic Stenosis"
    else:
        return "Normal"

# ------------------ Save and View Case History ------------------
def save_case(name, age, gender, date, diagnosis):
    st.session_state.cases.append({
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "age": age,
        "gender": gender,
        "date": date,
        "diagnosis": diagnosis
    })

# ------------------ Analysis Logic ------------------
if submit_info and uploaded_file is not None:
    st.subheader("📊 PCG Analysis Results")
    y, sr = librosa.load(uploaded_file)
    y_denoised = denoise_audio(y, sr)
    display_waveform(y_denoised, sr)
    features = extract_features(y_denoised, sr)
    result = dummy_rvhd_model(features)

    st.success(f"🩺 AI Diagnosis: **{result}**")
    save_case(name, age, gender, date, result)

elif uploaded_file is None and submit_info:
    st.warning("Please upload a PCG file to proceed with analysis.")

# ------------------ History Tab ------------------
st.subheader("📁 Previous Patient Cases")
if len(st.session_state.cases) == 0:
    st.info("No cases available yet.")
else:
    for case in st.session_state.cases[::-1]:
        with st.expander(f"🧾 {case['name']} ({case['date']})"):
            st.markdown(f"**ID:** {case['id']}")
            st.markdown(f"**Age:** {case['age']}")
            st.markdown(f"**Gender:** {case['gender']}")
            st.markdown(f"**Diagnosis:** {case['diagnosis']}")
    
