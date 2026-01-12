from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views
from .api_views import AudioClipView, AlertResetView, BaselineResetView, DeviceHandshakeView, DeviceTTSView, DeviceVisionImageView, EnvironmentReadingView, EnvironmentGuideConfirmView, MonitorSummaryView, MonitorTrendsView, STTView, VisionFrameView
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
    path('monitor/', views.monitor_dashboard, name='monitor_dashboard'),
    path('assistant/', views.assistant_ui, name='assistant_ui'),
    path('settings/ai/', views.ai_settings, name='ai_settings'),
    path('api/vision-frame/', VisionFrameView.as_view(), name='vision_frame'),
    path('api/audio/', AudioClipView.as_view(), name='audio_clip'),
    path('api/stt/', STTView.as_view(), name='stt_api'),
    path('api/device-handshake/', DeviceHandshakeView.as_view(), name='device_handshake'),
    path('api/environment/', EnvironmentReadingView.as_view(), name='environment_reading'),
    path('api/environment/confirm/', EnvironmentGuideConfirmView.as_view(), name='environment_confirm'),
    path('api/alerts/reset/', AlertResetView.as_view(), name='alerts_reset'),
    path('api/device/<int:device_id>/vision-frame/', DeviceVisionImageView.as_view(), name='device_vision_frame'),
    path('api/monitor/summary/', MonitorSummaryView.as_view(), name='monitor_summary'),
    path('api/monitor/trends/', MonitorTrendsView.as_view(), name='monitor_trends'),
    path('api/baseline/reset/', BaselineResetView.as_view(), name='baseline_reset'),
    path('api/tts/', DeviceTTSView.as_view(), name='device_tts'),
    path('api/', include(router.urls)),
]


