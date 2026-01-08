from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path("ws/vision/<int:device_id>/", consumers.VisionConsumer.as_asgi()),
]
