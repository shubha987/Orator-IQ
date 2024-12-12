import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import base64
from datetime import datetime

# Function to save video
def save_video(video_data, filename):
    with open(filename, 'wb') as f:
        f.write(video_data)

# Function to convert video to base64
def video_to_base64(video_path):
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()
    return base64.b64encode(video_data).decode('utf-8')

# Streamlit app
st.title("AI Interview - Video & Audio Recorder")

# Select media type
media_type = st.selectbox("Select what you want to record:", ["Choose an option", "Video", "Audio"])

if media_type == "Video":
    st.header("Record Video")
    st.write("Click the 'Start Recording' button to start recording video.")

    # Create a temporary file to save the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = temp_file.name

    # Start recording button
    if st.button("Start Recording"):
        st.write("Recording...")

        # Capture video from webcam
        cap = cv2.VideoCapture("http://192.168.32.1:8080/?action=stream")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

        while True:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                st.image(frame, channels="BGR")
            else:
                break

            if st.button("Stop Recording"):
                break

        cap.release()
        out.release()
        st.write("Recording stopped.")

        # Save video to server
        video_data = open(video_path, 'rb').read()
        save_video(video_data, video_path)
        st.write("Video saved.")

        # Display video
        st.video(video_path)

        # Convert video to base64
        video_base64 = video_to_base64(video_path)
        st.write("Video in base64 format:")
        st.text(video_base64)

elif media_type == "Audio":
    st.header("Record Audio")
    st.write("Click the 'Start Recording' button to start recording audio.")

    # Create a temporary file to save the audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = temp_file.name

    # Start recording button
    if st.button("Start Recording"):
        st.write("Recording...")

        # Capture audio from microphone
        import sounddevice as sd
        from scipy.io.wavfile import write

        fs = 44100  # Sample rate
        seconds = 10  # Duration of recording

        st.write("Recording for 10 seconds...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write(audio_path, fs, myrecording)  # Save as WAV file

        st.write("Recording stopped.")

        # Save audio to server
        audio_data = open(audio_path, 'rb').read()
        save_video(audio_data, audio_path)
        st.write("Audio saved.")

        # Display audio
        st.audio(audio_path)

        # Convert audio to base64
        audio_base64 = video_to_base64(audio_path)
        st.write("Audio in base64 format:")
        st.text(audio_base64)