from django.db import models
from django.db.models import Q


class Elderly(models.Model):
    name = models.CharField(max_length=100)
    address = models.CharField(max_length=255, help_text="전주 지역구 포함")
    emergency_contact = models.CharField(max_length=100)
    baseline = models.JSONField(default=dict, blank=True)
    learning_progress = models.PositiveSmallIntegerField(default=0, help_text="0~100 진행률")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.name} ({self.address})"


class HealthLog(models.Model):
    elderly = models.ForeignKey(Elderly, on_delete=models.CASCADE, related_name="health_logs")
    face_score = models.FloatField()
    activity_level = models.FloatField()
    environment = models.JSONField(default=dict, blank=True, help_text="온습도 등 환경 데이터")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.elderly.name} {self.created_at:%Y-%m-%d %H:%M}"


class Alert(models.Model):
    LEVEL_RED = "RED"
    LEVEL_YELLOW = "YELLOW"
    LEVEL_CHOICES = [
        (LEVEL_RED, "RED"),
        (LEVEL_YELLOW, "YELLOW"),
    ]

    elderly = models.ForeignKey(Elderly, on_delete=models.CASCADE, related_name="alerts")
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES)
    risk_type = models.CharField(max_length=100, help_text="낙상 등 위험 내용")
    handled = models.BooleanField(default=False)
    skeleton_data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.elderly.name} {self.level} {self.risk_type}"


class Quest(models.Model):
    elderly = models.ForeignKey(
        Elderly, on_delete=models.SET_NULL, related_name="quests", null=True, blank=True
    )
    title = models.CharField(max_length=200)
    is_completed = models.BooleanField(default=False)
    badge_image = models.CharField(max_length=255, blank=True, help_text="배지 이미지 경로 또는 URL")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title


class Owner(models.Model):
    user_id = models.CharField(max_length=32, unique=True)
    name = models.CharField(max_length=100)
    age = models.PositiveSmallIntegerField(null=True, blank=True)
    conditions = models.CharField(max_length=255, blank=True)
    phone = models.CharField(max_length=30, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.user_id} {self.name}"


class Device(models.Model):
    class DetectionMode(models.TextChoices):
        NORMAL = "normal", "Normal"
        CARE = "care", "Care"
        EMERGENCY = "emergency", "Emergency"

    owner = models.ForeignKey(Owner, on_delete=models.CASCADE, related_name="devices")
    elderly = models.ForeignKey(
        Elderly,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="devices",
    )
    serial = models.CharField(max_length=64, unique=True)
    is_primary = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    detection_mode = models.CharField(
        max_length=16, choices=DetectionMode.choices, default=DetectionMode.NORMAL
    )
    alert_enabled = models.BooleanField(default=True)
    config = models.JSONField(default=dict, blank=True)
    last_seen_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["owner"],
                condition=Q(is_primary=True),
                name="unique_primary_device_per_owner",
            )
        ]

    def __str__(self) -> str:
        status = "primary" if self.is_primary else "test"
        return f"{self.serial} ({status})"


class DeviceToken(models.Model):
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name="tokens")
    token_hash = models.CharField(max_length=64, unique=True)
    revoked = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.device.serial} token"



class EnvironmentReading(models.Model):
    temperature = models.FloatField()
    humidity = models.FloatField()
    source = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
