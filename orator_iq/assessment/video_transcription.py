import cv2
import sounddevice as sd
import queue
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Global variables for real-time audio processing
audio_queue = queue.Queue()
samplerate = 16000  # Wav2Vec2 requires 16 kHz audio

# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def audio_callback(indata, frames, time, status):
    """
    Callback function to capture audio in real-time.
    """
    if status:
        print(status)
    audio_queue.put(indata.copy())

def transcribe_audio(audio_chunk):
    """
    Transcribe an audio chunk using Wav2Vec2.
    """
    audio_chunk = audio_chunk.flatten()
    # Process the audio chunk using the processor
    inputs = processor(audio_chunk, sampling_rate=samplerate, return_tensors="pt", padding=True)
    
    # Run the model inference with the correct input values
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # Corrected here
    
    # Get the predicted tokens from the model
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode the tokens into human-readable transcription
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def main():
    # Open the Camera 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Start Audio Stream
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        print("Recording video and audio. Press 'q' to stop.")
        
        while True:
            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Real-Time Transcription", frame)
            
            # Process audio chunks for transcription
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                try:
                    transcription = transcribe_audio(audio_chunk)
                    print("Transcription:", transcription)
                except Exception as e:
                    print(f"Error during transcription: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
