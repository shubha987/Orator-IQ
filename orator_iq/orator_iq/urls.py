from django.urls import path
from . import views

urlpatterns = [
    path('record-video-transcribe/', views.record_video_and_transcribe_audio, name='record_video_and_transcribe_audio'),
]
