from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from .models import User, Video
from datetime import datetime

def home(request):
    return render(request, 'landing.html')

def pricing(request):
    return render(request, 'pricing.html')

@csrf_protect
def register(request):
    if request.method == 'POST':
        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        email = request.POST.get('email')
        password = request.POST.get('password')
        repassword = request.POST.get('repassword')
        if password != repassword:
            return HttpResponse('Your passwords did not match')
        my_user = User.objects.create_user(
            email=email,
            password=password,
            firstname=fname,
            lastname=lname
        )
        my_user.save()
        print(f"User created: {my_user}")
        return redirect('login')
    return render(request, 'signup.html')

@csrf_exempt
def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        print(f"Form data: email={email}, password={password}")
        print(f"Attempting to authenticate user: {email} with password: {password}")
        user_model = authenticate(request, username=email, password=password)
        print("Authenticated User:", user_model)
        if user_model is not None:
            auth_login(request, user_model)
            return redirect('profile')
        else:
            return HttpResponse("Email or password is incorrect!!!")
    return render(request, 'login.html')

@login_required
def profile(request):
    return render(request, 'student_dashboard.html')

@login_required
@csrf_exempt
def save_video(request):
    if request.method == 'POST':
        video_file = request.FILES['video']
        user = request.user
        video = Video(user=user, video=video_file, date=datetime.now())
        video.save()
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'failed'})

def video(request):
    return render(request, 'video.html')