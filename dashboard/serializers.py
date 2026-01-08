from rest_framework import serializers

from .models import Alert, Device, Elderly, HealthLog, Owner, Quest


class ElderlySerializer(serializers.ModelSerializer):
    class Meta:
        model = Elderly
        fields = [
            "id",
            "name",
            "address",
            "emergency_contact",
            "baseline",
            "learning_progress",
            "created_at",
        ]


class HealthLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = HealthLog
        fields = [
            "id",
            "elderly",
            "face_score",
            "activity_level",
            "environment",
            "created_at",
        ]


class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = [
            "id",
            "elderly",
            "level",
            "risk_type",
            "handled",
            "skeleton_data",
            "created_at",
        ]


class QuestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Quest
        fields = [
            "id",
            "elderly",
            "title",
            "is_completed",
            "badge_image",
            "created_at",
            "updated_at",
        ]


class OwnerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Owner
        fields = [
            "id",
            "user_id",
            "name",
            "age",
            "conditions",
            "phone",
            "created_at",
            "updated_at",
        ]


class DeviceSerializer(serializers.ModelSerializer):
    owner_user_id = serializers.CharField(source="owner.user_id", read_only=True)
    elderly_name = serializers.CharField(source="elderly.name", read_only=True)

    class Meta:
        model = Device
        fields = [
            "id",
            "owner",
            "owner_user_id",
            "serial",
            "is_primary",
            "is_active",
            "detection_mode",
            "alert_enabled",
            "config",
            "elderly",
            "elderly_name",
            "last_seen_at",
            "created_at",
            "updated_at",
        ]
