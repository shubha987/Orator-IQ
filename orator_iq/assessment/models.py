from django.db import models

class VideoTranscription(models.Model):
    """
    Model to store video files, audio files, and the corresponding transcription.
    """
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    audio_file = models.FileField(upload_to='audio/', null=True, blank=True)
    transcription = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)  # Automatically set when the object is created

    def __str__(self):
        return f"VideoTranscription {self.id} - {self.transcription[:50]}"  # Display first 50 chars of transcription

    class Meta:
        ordering = ['-created_at']  # Optionally order by creation date, most recent first
