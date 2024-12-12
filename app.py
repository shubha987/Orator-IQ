import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import base64
from datetime import datetime
import whisper
import language_tool_python
from nltk.corpus import cmudict
from pydub import AudioSegment, silence
import sounddevice as sd
from scipy.io.wavfile import write
import noisereduce as nr
import librosa
import soundfile as sf

# Function to save video
def save_video(video_data, filename):
    with open(filename, 'wb') as f:
        f.write(video_data)

# Function to convert video to base64
def video_to_base64(video_path):
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()
    return base64.b64encode(video_data).decode('utf-8')

# Function to extract audio from video
def extract_audio(video_file, output_audio_file):
    command = f"ffmpeg -i {video_file} -q:a 0 -map a {output_audio_file} -y"
    os.system(command)

# Function to reduce noise in audio
def reduce_noise(audio_file, output_audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    sf.write(output_audio_file, reduced_noise, sr)

# Function to transcribe audio to text
def real_time_transcription(audio_file):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file)
    transcription = model.transcribe(audio)
    return transcription["text"]

# Function to analyze grammar
def grammar_analysis(text):
    tool = language_tool_python.LanguageTool('en-IN')
    matches = tool.check(text)
    errors = [match.ruleId for match in matches]
    suggestions = [match.message for match in matches]
    return errors, suggestions

# Function to check pronunciation
def check_pronunciation(word):
    pronunciation = cmudict.dict()
    return pronunciation.get(word.lower(), "Word not found in dictionary")

# Function to calculate speaking rate
def speaking_rate_from_speaking_segments(transcribed_text, audio_file):
    audio = AudioSegment.from_wav(audio_file)
    non_silent_ranges = silence.detect_nonsilent(audio, min_silence_len=700, silence_thresh=-40)
    total_speaking_duration = sum((end - start) for start, end in non_silent_ranges) / 1000.0
    words = len(transcribed_text.split())
    speaking_rate = words / (total_speaking_duration / 60) if total_speaking_duration > 0 else 0
    return speaking_rate, total_speaking_duration

# Function to analyze pauses
def pause_analysis_during_speech(audio_file, transcription, min_silence_len=500, silence_thresh=-40):
    audio = AudioSegment.from_wav(audio_file)
    non_silent_ranges = silence.detect_nonsilent(audio, min_silence_len=700, silence_thresh=silence_thresh)
    speaking_segments = non_silent_ranges
    total_pauses = 0
    pause_details = []
    for start, end in speaking_segments:
        segment_audio = audio[start:end]
        pauses = silence.detect_silence(segment_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        pauses = [(start + pause_start, start + pause_end) for pause_start, pause_end in pauses]
        total_pauses += len(pauses)
        pause_details.extend(pauses)
    return total_pauses, pause_details

# Function to count filler words
def filler_word_usage(text):
    fillers = ["um", "uh", "like", "you know", "sort of"]
    filler_count = {filler: text.lower().count(filler) for filler in fillers}
    return filler_count

# Function to calculate SNR
def calculate_snr(audio_file):
    audio = AudioSegment.from_file(audio_file)
    samples = np.array(audio.get_array_of_samples())
    signal = np.mean(samples**2)
    noise = np.var(samples)
    snr = 10 * np.log10(signal / noise)
    return snr

# Streamlit app
st.title("AI Interview - Video & Audio Recorder")

# Select media type
media_type = st.selectbox("Select what you want to record:", ["Choose an option", "Video", "Audio"], key="media_selector")

if media_type == "Video":
    st.header("Record Video")
    st.write("Click the 'Start Recording' button to start recording video.")

    # Create a temporary file to save the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = temp_file.name

    # Recording state
    if 'recording' not in st.session_state:
        st.session_state.recording = False

    # Start/Stop recording buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Recording", key="start_video", disabled=st.session_state.recording):
            st.session_state.recording = True
            st.write("Recording...")
            
    with col2:
        if st.button("Stop Recording", key="stop_video", disabled=not st.session_state.recording):
            st.session_state.recording = False
            st.write("Recording stopped")

    if st.session_state.recording:
        stframe = st.empty()
        cap = cv2.VideoCapture("http://192.168.32.1:8080/?action=stream")
        
        if not cap.isOpened():
            st.error("Could not open video device")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (640,480))
            
            while st.session_state.recording:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    stframe.image(frame, channels="BGR")
                else:
                    break
                    
            cap.release()
            out.release()

            # Process the recorded video
            if os.path.exists(video_path):
                st.video(video_path)
                
                # Extract and process audio
                audio_file = "response_audio.wav"
                extract_audio(video_path, audio_file)
                
                if os.path.exists(audio_file):
                    processed_audio_file = "processed_audio.wav"
                    reduce_noise(audio_file, processed_audio_file)
                    
                    with st.spinner('Analyzing video...'):
                        # Transcribe audio to text
                        transcription = real_time_transcription(processed_audio_file)
                        st.subheader("Analysis Results")
                        st.write("Transcription:", transcription)
                        
                        # Analysis results in expandable sections
                        with st.expander("Grammar Analysis"):
                            errors, suggestions = grammar_analysis(transcription)
                            st.write("Errors:", errors)
                            st.write("Suggestions:", suggestions)
                            
                        with st.expander("Speech Analysis"):
                            speaking_rate, duration = speaking_rate_from_speaking_segments(transcription, processed_audio_file)
                            st.write(f"Speaking Rate: {speaking_rate:.2f} WPM")
                            st.write(f"Duration: {duration:.2f} seconds")
                            
                            total_pauses, pause_details = pause_analysis_during_speech(processed_audio_file, transcription)
                            st.write(f"Total Pauses: {total_pauses}")
                            
                            filler_count = filler_word_usage(transcription)
                            st.write("Filler Words:", filler_count)
                            
                            snr = calculate_snr(processed_audio_file)
                            st.write(f"Audio Quality (SNR): {snr:.2f} dB")

elif media_type == "Audio":
    st.header("Record Audio")
    st.write("Click the 'Start Recording' button to start recording audio.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = temp_file.name

    if st.button("Start Recording", key="start_audio"):
        with st.spinner('Recording audio for 10 seconds...'):
            fs = 44100
            duration = 10
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
            sd.wait()
            write(audio_path, fs, recording)
            
            st.audio(audio_path)
            
            with st.spinner('Analyzing audio...'):
                processed_audio_file = "processed_audio.wav"
                reduce_noise(audio_path, processed_audio_file)
                
                transcription = real_time_transcription(processed_audio_file)
                st.subheader("Analysis Results")
                st.write("Transcription:", transcription)
                
                with st.expander("Analysis Details"):
                    speaking_rate, duration = speaking_rate_from_speaking_segments(transcription, processed_audio_file)
                    st.write(f"Speaking Rate: {speaking_rate:.2f} WPM")
                    st.write(f"Duration: {duration:.2f} seconds")
                    
                    filler_count = filler_word_usage(transcription)
                    st.write("Filler Words:", filler_count)