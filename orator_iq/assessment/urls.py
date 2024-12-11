##urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='assessment-home'),
    path('pricing/', views.pricing, name='assessment-pricing'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('profile/', views.profile, name='profile'),
    path('video/', views.video, name='video'),
    path('save_video/', views.save_video, name='save_video'),
]
