from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'/video/', consumers.VideoConsumer.as_asgi()),
]