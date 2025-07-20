import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import io
import scipy.signal
from scipy.io.wavfile import write
import tempfile

st.set_page_config(page_title="PCG Audio Feature Analysis", layout="wide")

def preprocess_audio(audio, sr, uid):
    st.subheader("Preprocessing")

    # Noise reduction
    st.write("🔉 Applying noise reduction...")
    amplitude_factor = st.slider(
        "Amplitude scaling",
        0.1, 5.0, 1.0,
        key=f"amplitude_slider_{uid}"
    )
    audio = audio * amplitude_factor

    noise_threshold = st.slider(
        "Noise threshold",
        0.0, 0.1, 0.01,
        step=0.005,
        key=f"noise_threshold_slider_{uid}"
    )
    audio[np.abs(audio) < noise_threshold] = 0

    # Normalization
    st.write("📏 Normalizing audio...")
    audio = librosa.util.normalize(audio)

    return audio

def extract_features(audio, sr):
    st.subheader("🎼 Feature Extraction")

    st.write("🔍 Extracting MFCCs...")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    st.write("MFCC shape:", mfccs.shape)

    st.write("🎵 Extracting Chroma Features...")
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    st.write("Chroma shape:", chroma.shape)

    st.write("⚡ Extracting Spectral Contrast...")
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    st.write("Spectral Contrast shape:", contrast.shape)

    return mfccs, chroma, contrast

def plot_waveform(audio, sr):
    st.subheader("📈 Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    st.pyplot(fig)

def plot_spectrogram(audio, sr):
    st.subheader("🎨 Spectrogram")
    X = librosa.stft(audio)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    st.pyplot(fig)

def plot_mfccs(mfccs):
    st.subheader("🎚 MFCC")
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

def save_processed_audio(audio, sr):
    st.subheader("💾 Download Processed Audio")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV')
    st.download_button(
        label="Download Processed Audio",
        data=buf,
        file_name="processed_audio.wav",
        mime="audio/wav",
        key="download_button"
    )

def analyze_audio(uploaded_file):
    st.subheader("🔬 Audio Analysis")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Generate unique ID from file name
        uid = uploaded_file.name.replace(".", "_")

        # Load file
        audio, sr = librosa.load(uploaded_file, sr=None)
        st.write("Sample rate:", sr)
        st.write("Audio duration:", len(audio) / sr, "seconds")

        # Preprocessing
        audio = preprocess_audio(audio, sr, uid)

        # Feature extraction
        mfccs, chroma, contrast = extract_features(audio, sr)

        # Visualizations
        plot_waveform(audio, sr)
        plot_spectrogram(audio, sr)
        plot_mfccs(mfccs)

        # Save and offer download
        save_processed_audio(audio, sr)

# Streamlit App
st.title("💡 PCG Audio Feature Analysis using AyuSynk")

st.write("""
Upload a phonocardiogram (PCG) WAV file recorded via AyuSynk, and this app will process it for
denoising, amplitude adjustment, and extract audio features like MFCC, Chroma, and Spectrogram.
""")

uploaded_file = st.file_uploader("📤 Upload your WAV file", type=["wav"], key="file_uploader")

if uploaded_file is not None:
    analyze_audio(uploaded_file)
