
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import io

st.set_page_config(page_title="Heart Sound Analyzer", layout="centered")

st.title("🫀 Heart Sound Analyzer (PCG)")
st.markdown("Upload your **heart sound (.wav)** file recorded from AyuSynk or any digital stethoscope.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    # Read the file
    samplerate, data = wav.read(uploaded_file)
    st.audio(uploaded_file, format='audio/wav')
    
    st.write(f"**Sample Rate:** {samplerate} Hz")
    st.write(f"**Duration:** {len(data) / samplerate:.2f} seconds")
    
    # If stereo, take one channel
    if len(data.shape) == 2:
        data = data[:, 0]

    # Plot waveform
    st.subheader("📉 Waveform")
    time = np.linspace(0, len(data) / samplerate, num=len(data))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time, data, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Heart Sound Waveform")
    st.pyplot(fig)
    
    st.markdown("✅ *Basic analysis complete.*\n\n🔬 **Next step:** Add murmur detection (AI coming soon).")
else:
    st.info("Please upload a .wav file to proceed.")
