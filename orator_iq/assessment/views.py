# views.py
import os
from django.http import JsonResponse
from django.shortcuts import render
from .models import VideoTranscription
from .video_transcription import record_and_transcribe_video_audio  # Import the function

def record_video_and_transcribe_audio(request):
    if request.method == 'POST':
        # Set video and audio file paths
        video_file_path = 'media/videos/recorded_video.avi'
        audio_file_path = 'media/audio/extracted_audio.wav'

        try:
            # Call the function from video_transcription.py to record and transcribe
            transcription = record_and_transcribe_video_audio(video_file_path, audio_file_path)
            
            # Save the video, audio, and transcription in the database
            video_record = VideoTranscription.objects.create(
                video_file=video_file_path,
                audio_file=audio_file_path,
                transcription=transcription
            )

            # Return the response with the transcription and file URLs
            return JsonResponse({
                "video_url": video_file_path,
                "audio_url": audio_file_path,
                "transcription": transcription
            })
        
        except Exception as e:
            return JsonResponse({"error": f"An error occurred: {e}"}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=400)
