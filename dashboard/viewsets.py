import json
import os

from rest_framework import status, viewsets
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .models import Alert, Device, Elderly, HealthLog, Owner, Quest
from .serializers import (
    AlertSerializer,
    DeviceSerializer,
    ElderlySerializer,
    HealthLogSerializer,
    OwnerSerializer,
    QuestSerializer,
)


class ElderlyViewSet(viewsets.ModelViewSet):
    queryset = Elderly.objects.all()
    serializer_class = ElderlySerializer
    authentication_classes = []
    permission_classes = [AllowAny]


class OwnerViewSet(viewsets.ModelViewSet):
    queryset = Owner.objects.all()
    serializer_class = OwnerSerializer
    authentication_classes = []
    permission_classes = [AllowAny]


class DeviceViewSet(viewsets.ModelViewSet):
    queryset = Device.objects.select_related("owner", "elderly").all()
    serializer_class = DeviceSerializer
    authentication_classes = []
    permission_classes = [AllowAny]


class HealthLogViewSet(viewsets.ModelViewSet):
    queryset = HealthLog.objects.select_related("elderly").all()
    serializer_class = HealthLogSerializer
    authentication_classes = []
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        data = self._coerce_request_data(request.data)
        elderly_id = self._ensure_elderly(data.get("elderly"))
        data["elderly"] = elderly_id
        self._normalize_environment(data)
        self._ensure_metrics(data)

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    @staticmethod
    def _coerce_request_data(raw_data):
        if hasattr(raw_data, "dict"):
            return raw_data.dict()
        return dict(raw_data)

    @staticmethod
    def _coerce_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _ensure_elderly(self, elderly_value):
        elderly_id = self._coerce_int(elderly_value)
        if not elderly_id:
            default_id = self._coerce_int(os.getenv("DEFAULT_ELDERLY_ID"))
            elderly_id = default_id or 1

        Elderly.objects.get_or_create(
            id=elderly_id,
            defaults={
                "name": f"미등록 어르신 {elderly_id}",
                "address": "전주시 미등록",
                "emergency_contact": "미등록",
                "baseline": {},
                "learning_progress": 0,
            },
        )
        return elderly_id

    @staticmethod
    def _normalize_environment(data):
        environment = data.get("environment")
        if isinstance(environment, str):
            try:
                environment = json.loads(environment)
            except json.JSONDecodeError:
                environment = {}
        if not isinstance(environment, dict):
            environment = {}

        temperature = data.pop("temperature", None)
        humidity = data.pop("humidity", None)
        if temperature is not None and "temperature" not in environment:
            environment["temperature"] = temperature
        if humidity is not None and "humidity" not in environment:
            environment["humidity"] = humidity

        data["environment"] = environment

    @staticmethod
    def _ensure_metrics(data):
        for field in ("face_score", "activity_level"):
            if data.get(field) in (None, ""):
                data[field] = 0.0


class AlertViewSet(viewsets.ModelViewSet):
    queryset = Alert.objects.select_related("elderly").all()
    serializer_class = AlertSerializer
    authentication_classes = []
    permission_classes = [AllowAny]


class QuestViewSet(viewsets.ModelViewSet):
    queryset = Quest.objects.select_related("elderly").all()
    serializer_class = QuestSerializer
    authentication_classes = []
    permission_classes = [AllowAny]
