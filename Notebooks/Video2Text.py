import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa


# Load pre-trained Wav2Vec 2.0 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Set the sampling rate (should be 16 kHz for Wav2Vec 2.0)
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Size of each audio chunk

# Initialize a list to hold the audio data in real-time
audio_data = []

def audio_callback(indata, frames, time, status):
    """ Callback function to handle real-time audio data """
    if status:
        print(status, flush=True)
    # Append the audio data chunk to the list
    audio_data.append(indata.copy())

def transcribe_audio():
    """ Function to transcribe the audio data in real-time """
    global audio_data
    # Concatenate all recorded chunks into one audio array
    audio_input = np.concatenate(audio_data, axis=0)
    
    # Resample the audio to the required sampling rate (16 kHz)
    audio_input = librosa.resample(audio_input, orig_sr=SAMPLE_RATE, target_sr=16000)

    # Preprocess the audio for Wav2Vec 2.0
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)

    # Perform inference with Wav2Vec 2.0
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    # Get predicted ids and decode to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcribed_text = processor.decode(predicted_ids[0])

    # Output the transcribed text
    print(f"Transcribed Text: {transcribed_text}")

    # Clear the audio data list to start fresh for the next chunk
    audio_data = []

# Start the audio stream in real-time
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE)
stream.start()

try:
    while True:
        # Run the transcription function every second
        transcribe_audio()
        # Sleep or wait before processing the next chunk (you can adjust the wait time as needed)
        time.sleep(1)

except KeyboardInterrupt:
    # Stop the audio stream when interrupted
    print("Stopping real-time transcription.")
    stream.stop()
