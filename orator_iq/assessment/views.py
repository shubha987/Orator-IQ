import subprocess
import noisereduce as nr
import librosa
import soundfile as sf
import whisper
import language_tool_python
from pydub import AudioSegment, silence
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse

def extract_audio(video_file, output_audio_file):
    command = f"ffmpeg -i {video_file} -q:a 0 -map a {output_audio_file} -y"
    subprocess.run(command, shell=True, check=True)

def real_time_transcription(audio_file):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file)
    transcription = model.transcribe(audio)
    return transcription["text"]

def grammar_analysis(text):
    tool = language_tool_python.LanguageTool('en-IN')
    matches = tool.check(text)
    errors = [match.ruleId for match in matches]
    suggestions = [match.message for match in matches]
    return errors, suggestions

def speaking_rate(transcribed_text, duration_seconds):
    words = len(transcribed_text.split())
    rate = words / (duration_seconds / 60)  # Words per minute
    return rate

def pause_analysis(audio_file):
    audio = AudioSegment.from_wav(audio_file)
    pauses = silence.detect_silence(audio, min_silence_len=500, silence_thresh=-40)
    return len(pauses), pauses

def filler_word_usage(text):
    fillers = ["um", "uh", "like", "you know", "sort of"]
    filler_count = {filler: text.lower().count(filler) for filler in fillers}
    return filler_count

def calculate_snr(audio_file):
    audio = AudioSegment.from_file(audio_file)
    samples = np.array(audio.get_array_of_samples())
    signal = np.mean(samples**2)
    noise = np.var(samples)
    snr = 10 * np.log10(signal / noise)
    return snr

def generate_feedback(transcription, audio_file, duration):
    # Grammar
    errors, suggestions = grammar_analysis(transcription)
    
    # Speaking Rate
    rate = speaking_rate(transcription, duration)
    
    # Pause Patterns
    num_pauses, pauses = pause_analysis(audio_file)
    
    # Filler Word Usage
    fillers = filler_word_usage(transcription)
    
    # Voice Clarity
    snr = calculate_snr(audio_file)
    
    feedback = {
        "Grammar Errors": errors,
        "Grammar Suggestions": suggestions,
        "Speaking Rate (WPM)": rate,
        "Number of Pauses": num_pauses,
        "Pause Details (ms)": pauses,
        "Filler Word Usage": fillers,
        "Voice Clarity (SNR in dB)": snr,
    }
    return feedback

def home(request):
    if request.method == 'POST' and request.FILES['video_file']:
        # Get the uploaded video
        video_file = request.FILES['video_file']
        
        # Save video to a temporary file
        with open("temp_video.mp4", 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)
        
        # Extract audio from the video
        extract_audio("temp_video.mp4", "response_audio.wav")
        
        # Apply noise reduction and process audio
        audio, sr = librosa.load("response_audio.wav", sr=16000)
        reduced_noise = nr.reduce_noise(y=audio, sr=sr)
        sf.write("processed_audio.wav", reduced_noise, sr)
        
        # Perform transcription
        transcription = real_time_transcription("processed_audio.wav")
        
        # Generate feedback
        feedback = generate_feedback(transcription, "processed_audio.wav", duration=45)
        
        return render(request, 'assessment/home.html', {'feedback': feedback, 'transcription': transcription})
    
    return render(request, 'assessment/home.html')
