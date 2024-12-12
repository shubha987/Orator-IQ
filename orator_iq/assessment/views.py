import subprocess
import noisereduce as nr
import librosa
import soundfile as sf
import whisper
import language_tool_python
from pydub import AudioSegment, silence
import numpy as np
from nltk.corpus import cmudict
from django.shortcuts import render
from django.http import JsonResponse

# Extract audio from the uploaded video
def extract_audio(video_file, output_audio_file):
    command = f"ffmpeg -i {video_file} -q:a 0 -map a {output_audio_file} -y"
    subprocess.run(command, shell=True, check=True)

# Real-time transcription of audio using Whisper model
def real_time_transcription(audio_file):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file)
    transcription = model.transcribe(audio)
    return transcription["text"]

# Grammar analysis to find errors and suggestions
def grammar_analysis(text):
    tool = language_tool_python.LanguageTool('en-IN')
    matches = tool.check(text)
    errors = [match.ruleId for match in matches]
    suggestions = [match.message for match in matches]
    return errors, suggestions

# Speaking rate calculation based on speaking segments
def speaking_rate_from_speaking_segments(transcribed_text, audio_file):
    """
    Calculate speaking rate based on speaking segments in the audio.
    
    :param transcribed_text: Transcribed text of the candidate's speech.
    :param audio_file: Path to the audio file.
    :return: Speaking rate (WPM) and total speaking time (seconds).
    """
    audio = AudioSegment.from_wav(audio_file)
    # Detect non-silent segments (candidate speaking)
    non_silent_ranges = silence.detect_nonsilent(audio, min_silence_len=700, silence_thresh=-40)
    # Calculate total speaking time
    total_speaking_duration = sum((end - start) for start, end in non_silent_ranges) / 1000.0  # in seconds
    # Calculate speaking rate (WPM)
    words = len(transcribed_text.split())
    speaking_rate = words / (total_speaking_duration / 60) if total_speaking_duration > 0 else 0
    return speaking_rate, total_speaking_duration

# Analyze pauses during the speech (while speaking)
def pause_analysis_during_speech(audio_file, transcription):
    """
    Analyze pauses during candidate speaking time, excluding silence outside speaking segments.
    
    :param audio_file: Path to the audio file.
    :param transcription: Transcribed text of the candidate's speech.
    :return: Number of pauses and details of pause durations (start, end).
    """
    audio = AudioSegment.from_wav(audio_file)
    # Detect non-silent segments (candidate speaking)
    non_silent_ranges = silence.detect_nonsilent(audio, min_silence_len=700, silence_thresh=-40)
    
    # Analyze pauses within speaking segments
    total_pauses = 0
    pause_details = []
    for start, end in non_silent_ranges:
        # Extract the candidate's speaking segment
        segment_audio = audio[start:end]
        # Detect pauses within this segment
        pauses = silence.detect_silence(segment_audio, min_silence_len=500, silence_thresh=-40)
        # Adjust pause times relative to the original audio
        pauses = [(start + pause_start, start + pause_end) for pause_start, pause_end in pauses]
        total_pauses += len(pauses)
        pause_details.extend(pauses)
    
    return total_pauses, pause_details

# Filler word usage analysis (common filler words in speech)
def filler_word_usage(text):
    fillers = ["um", "uh", "like", "you know", "sort of"]
    filler_count = {filler: text.lower().count(filler) for filler in fillers}
    return filler_count

# Calculate Signal-to-Noise Ratio (SNR) for audio clarity
def calculate_snr(audio_file):
    audio = AudioSegment.from_file(audio_file)
    samples = np.array(audio.get_array_of_samples())
    signal = np.mean(samples**2)
    noise = np.var(samples)
    snr = 10 * np.log10(signal / noise)
    return snr

# Pronunciation analysis using CMU Pronouncing Dictionary and comparing to expected pronunciation
def check_pronunciation_accuracy(transcribed_word, correct_pronunciation):
    """
    Compares the transcribed word's pronunciation to the expected pronunciation.
    
    :param transcribed_word: The word as transcribed in the speech.
    :param correct_pronunciation: Correct phonetic pronunciation from CMU dictionary.
    :return: Pronunciation accuracy feedback.
    """
    transcribed_phonemes = cmudict.dict().get(transcribed_word.lower())
    if not transcribed_phonemes:
        return "Word not found in dictionary"
    
    # Compare phonetic transcriptions (could be more sophisticated depending on your needs)
    accuracy = "Accurate" if transcribed_phonemes == correct_pronunciation else "Inaccurate"
    return accuracy

# Update pronunciation analysis to include accuracy checking for each word
def check_pronunciation(word):
    pronunciation = cmudict.dict()
    word_pronunciation = pronunciation.get(word.lower(), None)
    if word_pronunciation:
        # Checking first pronunciation variant
        return word_pronunciation[0]
    return "Word not found in dictionary"

# Pronunciation analysis for multiple words (return feedback on each word)
def pronunciation_analysis(transcribed_text):
    words = transcribed_text.split()[:5]  # Analyze the first 5 words for pronunciation
    pronunciation_feedback = {}
    
    for word in words:
        expected_pronunciation = check_pronunciation(word)
        pronunciation_accuracy = check_pronunciation_accuracy(word, expected_pronunciation)
        pronunciation_feedback[word] = {
            "Expected Pronunciation": expected_pronunciation,
            "Pronunciation Accuracy": pronunciation_accuracy
        }
    
    return pronunciation_feedback

# Modify the generate_feedback function to include pronunciation accuracy
def generate_feedback(transcription, audio_file):
    """
    Generate detailed feedback for the candidate's speech.
    
    :param transcription: Transcribed text of the candidate's speech.
    :param audio_file: Path to the processed audio file.
    :return: Dictionary containing feedback details.
    """
    # Grammar Analysis
    errors, suggestions = grammar_analysis(transcription)

    # Speaking Rate and Speaking Time
    speaking_rate, speaking_time = speaking_rate_from_speaking_segments(transcription, audio_file)

    # Pause Patterns (pauses while speaking)
    num_pauses, pauses = pause_analysis_during_speech(audio_file, transcription)

    # Filler Word Usage
    fillers = filler_word_usage(transcription)

    # Voice Clarity (SNR)
    snr = calculate_snr(audio_file)
    if snr < 10:
        voice_clarity = "Low"
    elif 10 <= snr < 20:
        voice_clarity = "Medium"
    else:
        voice_clarity = "High"

    # Pronunciation analysis (for first few words)
    pronunciation_feedback = pronunciation_analysis(transcription)

    # Feedback Compilation
    feedback = {
        "Grammar Errors": errors,
        "Grammar Suggestions": suggestions,
        "Speaking Rate (WPM)": speaking_rate,
        "Total Speaking Time (seconds)": speaking_time,
        "Number of Pauses": num_pauses,
        "Pause Details (ms)": pauses,
        "Filler Word Usage": fillers,
        "Voice Clarity": voice_clarity,
        "Pronunciation Feedback": pronunciation_feedback,
    }
    return feedback


# View to handle file uploads and feedback generation
def home(request):
    if request.method == 'POST' and request.FILES['video_file']:
        # Get the uploaded video file
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
        
        # Generate feedback based on transcription and audio file
        feedback = generate_feedback(transcription, "processed_audio.wav")
        
        # Render the feedback in the response
        return render(request, 'assessment/home.html', {'feedback': feedback, 'transcription': transcription})
    
    return render(request, 'assessment/home.html')
