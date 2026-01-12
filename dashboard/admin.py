from django.contrib import admin

from .models import AiSetting, Alert, Device, DeviceToken, Elderly, HealthLog, Owner, Quest


@admin.register(Elderly)
class ElderlyAdmin(admin.ModelAdmin):
    list_display = ("name", "address", "emergency_contact", "learning_progress", "created_at")
    search_fields = ("name", "address")


@admin.register(HealthLog)
class HealthLogAdmin(admin.ModelAdmin):
    list_display = ("elderly", "face_score", "activity_level", "created_at")
    list_filter = ("created_at",)


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("elderly", "level", "risk_type", "handled", "created_at")
    list_filter = ("level", "handled")


@admin.register(Quest)
class QuestAdmin(admin.ModelAdmin):
    list_display = ("title", "elderly", "is_completed", "created_at")
    list_filter = ("is_completed",)


@admin.register(Owner)
class OwnerAdmin(admin.ModelAdmin):
    list_display = ("user_id", "name", "age", "conditions", "phone", "created_at")
    search_fields = ("user_id", "name")


@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ("serial", "owner", "elderly", "is_primary", "is_active", "last_seen_at")
    list_filter = ("is_primary", "is_active", "detection_mode")
    search_fields = ("serial", "owner__user_id", "owner__name", "elderly__name")


@admin.register(DeviceToken)
class DeviceTokenAdmin(admin.ModelAdmin):
    list_display = ("device", "revoked", "created_at")
    list_filter = ("revoked",)


@admin.register(AiSetting)
class AiSettingAdmin(admin.ModelAdmin):
    list_display = ("id", "updated_at")
