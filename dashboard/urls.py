from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views
from .api_views import AudioClipView, DeviceHandshakeView, DeviceVisionImageView, EnvironmentReadingView, VisionFrameView
from .viewsets import (
    AlertViewSet,
    DeviceViewSet,
    ElderlyViewSet,
    HealthLogViewSet,
    OwnerViewSet,
    QuestViewSet,
)

router = DefaultRouter()
router.register(r"elderly", ElderlyViewSet)
router.register(r"owners", OwnerViewSet)
router.register(r"devices", DeviceViewSet)
router.register(r"healthlog", HealthLogViewSet)
router.register(r"alerts", AlertViewSet)
router.register(r"quests", QuestViewSet)

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('elderly/', views.elderly_list, name='elderly_list'),
    path('alerts/', views.alert_history, name='alert_history'),
    path('quests/', views.quest_center, name='quest_center'),
    path('environment/', views.environment_guide, name='environment_guide'),
    path('device/register/', views.device_register, name='device_register'),
    path('device/<int:device_id>/edit/', views.device_edit, name='device_edit'),
    path('device/<int:device_id>/vision/', views.device_vision, name='device_vision'),
    path('api/vision-frame/', VisionFrameView.as_view(), name='vision_frame'),
    path('api/audio/', AudioClipView.as_view(), name='audio_clip'),
    path('api/device-handshake/', DeviceHandshakeView.as_view(), name='device_handshake'),
    path('api/environment/', EnvironmentReadingView.as_view(), name='environment_reading'),
    path('api/device/<int:device_id>/vision-frame/', DeviceVisionImageView.as_view(), name='device_vision_frame'),
    path('api/', include(router.urls)),
]
