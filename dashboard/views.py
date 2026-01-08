from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List

import os

from django.db import transaction
from django.db.models import OuterRef, Subquery
from django.shortcuts import get_object_or_404, render
from django.utils import timezone

from .models import Alert, Device, Elderly, Owner, Quest

AREAS = ["노송동", "완산동", "덕진동", "효자동", "인후동"]


def extract_area(address: str) -> str:
    for area in AREAS:
        if area in address:
            return area
    parts = address.split()
    if len(parts) > 1:
        return parts[1]
    return "기타"


def generate_user_id() -> str:
    prefix = "U"
    last_owner = Owner.objects.order_by("-id").first()
    counter = 1
    if last_owner and last_owner.user_id.startswith(prefix):
        suffix = last_owner.user_id[len(prefix) :]
        if suffix.isdigit():
            counter = int(suffix) + 1

    while Owner.objects.filter(user_id=f"{prefix}{counter:04d}").exists():
        counter += 1

    return f"{prefix}{counter:04d}"


def latest_alert_subquery(field: str):
    latest = Alert.objects.filter(elderly=OuterRef("pk")).order_by("-created_at")
    return Subquery(latest.values(field)[:1])


def dashboard(request):
    elderly_qs = Elderly.objects.annotate(
        latest_level=latest_alert_subquery("level"),
        latest_risk=latest_alert_subquery("risk_type"),
        latest_alert_time=latest_alert_subquery("created_at"),
    )

    red_alerts = []
    region_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"safe": 0, "caution": 0, "danger": 0})

    for elder in elderly_qs:
        area = extract_area(elder.address)
        level = elder.latest_level
        baseline = elder.baseline or {}
        age = baseline.get("age", "-")
        if level == Alert.LEVEL_RED:
            risk_class = "danger"
            red_alerts.append(
                {
                    "name": elder.name,
                    "area": area,
                    "age": age,
                    "risk_reason": elder.latest_risk or "위험 감지",
                    "last_event": elder.latest_alert_time.strftime("%H:%M") if elder.latest_alert_time else "-",
                }
            )
        elif level == Alert.LEVEL_YELLOW:
            risk_class = "caution"
        else:
            risk_class = "safe"
        region_stats[area][risk_class] += 1

    region_labels = AREAS
    region_safe = [region_stats[area]["safe"] for area in region_labels]
    region_caution = [region_stats[area]["caution"] for area in region_labels]
    region_danger = [region_stats[area]["danger"] for area in region_labels]

    total_quests = Quest.objects.count()
    completed_quests = Quest.objects.filter(is_completed=True).count()
    mission_rate = int((completed_quests / total_quests) * 100) if total_quests else 0

    completed_by_area: Dict[str, int] = defaultdict(int)
    for quest in Quest.objects.filter(is_completed=True, elderly__isnull=False).select_related("elderly"):
        completed_by_area[extract_area(quest.elderly.address)] += 1
    top_area = max(completed_by_area, key=completed_by_area.get) if completed_by_area else "없음"

    return render(
        request,
        "dashboard/dashboard.html",
        {
            "title": "통합 관제",
            "page_name": "dashboard",
            "red_alerts": red_alerts,
            "region_chart": {
                "labels": region_labels,
                "safe": region_safe,
                "caution": region_caution,
                "danger": region_danger,
            },
            "mission_summary": {
                "completed": completed_quests,
                "total": total_quests,
                "rate": mission_rate,
                "top_area": top_area,
                "delayed": total_quests - completed_quests,
            },
            "mission_chart": {
                "completed": completed_quests,
                "remaining": max(total_quests - completed_quests, 0),
            },
        },
    )


def elderly_list(request):
    hour_labels = [f"{hour:02d}" for hour in range(24)]
    start = timezone.now() - timedelta(hours=24)
    elderly_profiles = []

    for elder in Elderly.objects.all():
        baseline = elder.baseline or {}
        current_day = int(baseline.get("current_day", 0))
        total_days = int(baseline.get("total_days", 7))
        percent = int((current_day / total_days) * 100) if total_days else elder.learning_progress

        latest_alert = elder.alerts.order_by("-created_at").first()
        if latest_alert and latest_alert.level == Alert.LEVEL_RED:
            risk_class, risk_label = "danger", "위험"
        elif latest_alert and latest_alert.level == Alert.LEVEL_YELLOW:
            risk_class, risk_label = "caution", "주의"
        else:
            risk_class, risk_label = "safe", "안정"

        activity_values = [0] * 24
        for log in elder.health_logs.filter(created_at__gte=start).order_by("created_at"):
            activity_values[log.created_at.hour] = log.activity_level

        elderly_profiles.append(
            {
                "id": elder.id,
                "name": elder.name,
                "initials": "".join([token[0] for token in elder.name.split()])[:2].upper() or "-",
                "age": baseline.get("age", "-"),
                "area": extract_area(elder.address),
                "address": elder.address,
                "emergency_contact": elder.emergency_contact,
                "conditions": baseline.get("conditions", "정보 없음"),
                "baseline_progress": {"current": current_day, "total": total_days, "percent": percent},
                "baseline_metrics": baseline.get("metrics", []),
                "activity": {"labels": hour_labels, "values": activity_values},
                "activity_script_id": f"activity-data-{elder.id}",
                "sleep": baseline.get("sleep", {"total": 0, "deep": 0, "disruptions": 0}),
                "risk_class": risk_class,
                "risk_label": risk_label,
            }
        )

    return render(
        request,
        "dashboard/elderly_list.html",
        {
            "title": "대상자 관리",
            "page_name": "elderly",
            "elderly_profiles": elderly_profiles,
        },
    )


def alert_history(request):
    base_points = [
        {"x": 60, "y": 12},
        {"x": 60, "y": 22},
        {"x": 45, "y": 30},
        {"x": 35, "y": 44},
        {"x": 75, "y": 30},
        {"x": 85, "y": 44},
        {"x": 95, "y": 60},
        {"x": 60, "y": 36},
        {"x": 50, "y": 54},
        {"x": 45, "y": 72},
        {"x": 70, "y": 54},
        {"x": 75, "y": 72},
        {"x": 80, "y": 86},
    ]
    alerts = []
    for alert in Alert.objects.select_related("elderly").all()[:20]:
        skeleton = alert.skeleton_data or {}
        points = skeleton.get("points") or base_points
        alerts.append(
            {
                "id": alert.id,
                "time": alert.created_at.strftime("%H:%M"),
                "type": alert.risk_type,
                "detail": skeleton.get("detail", alert.risk_type),
                "area": extract_area(alert.elderly.address),
                "handled": alert.handled,
                "status_label": "조치 완료" if alert.handled else "확인 필요",
                "validation_note": skeleton.get("note", "검증 필요"),
                "skeleton_points": points,
                "skeleton_script_id": f"skeleton-data-{alert.id}",
            }
        )

    return render(
        request,
        "dashboard/alert_history.html",
        {
            "title": "이상 징후 리포트",
            "page_name": "alerts",
            "alerts": alerts,
        },
    )


def quest_center(request):
    therapy_library = [
        {"title": "전주 객사길 1978", "meta": "노송동 옛 사진", "tag": "전주"},
        {"title": "가족 사진 1985", "meta": "완산동 가정 앨범", "tag": "가족"},
        {"title": "전주 영화제 거리", "meta": "추억 사진", "tag": "추억"},
    ]
    quest_suggestions = [quest.title for quest in Quest.objects.filter(is_completed=False)[:3]]
    if not quest_suggestions:
        quest_suggestions = [
            "옛날 전주역 이름 맞추기",
            "오늘의 날씨 설명하기",
            "내가 좋아하는 노래 한 소절",
        ]
    badges = [
        {"title": "노송동 깔끔왕", "criteria": "청결 미션 5회 완료", "icon": "🧹"},
        {"title": "전주 기억력 박사", "criteria": "회상 퀘스트 10회 완료", "icon": "🧠"},
        {"title": "걷기 챔피언", "criteria": "걷기 미션 연속 7일", "icon": "🚶"},
        {"title": "안전 지킴이", "criteria": "위험 알림 빠른 응답", "icon": "🛡️"},
    ]
    return render(
        request,
        "dashboard/quest_center.html",
        {
            "title": "콘텐츠 & 퀘스트",
            "page_name": "quests",
            "therapy_library": therapy_library,
            "quest_suggestions": quest_suggestions,
            "badges": badges,
        },
    )


def environment_guide(request):
    risk_households: List[Dict[str, Any]] = []
    temps: List[float] = []

    for elder in Elderly.objects.all():
        latest_log = elder.health_logs.exclude(environment={}).order_by("-created_at").first()
        if not latest_log:
            continue
        environment = latest_log.environment or {}
        temperature = environment.get("temperature")
        if temperature is not None:
            temps.append(float(temperature))
        if temperature is not None and float(temperature) >= 30:
            risk_households.append(
                {
                    "area": extract_area(elder.address),
                    "address": elder.address,
                    "risk_class": "danger",
                    "risk_label": "위험 가구",
                }
            )
        elif temperature is not None and float(temperature) <= 0:
            risk_households.append(
                {
                    "area": extract_area(elder.address),
                    "address": elder.address,
                    "risk_class": "caution",
                    "risk_label": "주의 필요",
                }
            )

    avg_temp = round(sum(temps) / len(temps), 1) if temps else None
    if avg_temp is None:
        weather_summary = {
            "title": "환경 데이터 없음",
            "detail": "온습도 센서 데이터가 아직 없습니다.",
            "temperature": "-",
        }
    elif avg_temp >= 30:
        weather_summary = {
            "title": "폭염 경보 지속",
            "detail": f"평균 온도 {avg_temp}°C",
            "temperature": avg_temp,
        }
    elif avg_temp <= 0:
        weather_summary = {
            "title": "한파 주의",
            "detail": f"평균 온도 {avg_temp}°C",
            "temperature": avg_temp,
        }
    else:
        weather_summary = {
            "title": "기온 안정",
            "detail": f"평균 온도 {avg_temp}°C",
            "temperature": avg_temp,
        }

    broadcast = {
        "current_message": "창문을 열고 시원한 바람을 느껴보세요.",
        "households": len(risk_households),
        "guides": [
            {
                "title": "물 자주 드시기",
                "detail": "2시간마다 물 한 컵 섭취를 안내합니다.",
            },
            {
                "title": "실내 환기",
                "detail": "10분마다 창문 개방 알림을 설정합니다.",
            },
        ],
    }

    return render(
        request,
        "dashboard/environment.html",
        {
            "title": "환경 가이드",
            "page_name": "environment",
            "weather_summary": weather_summary,
            "risk_households": risk_households,
            "broadcast": broadcast,
        },
    )


def device_register(request):
    context = {
        "title": "대상자 등록",
        "page_name": "device_vision",
    }

    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        age_raw = (request.POST.get("age") or "").strip()
        conditions = (request.POST.get("conditions") or "").strip()
        address = (request.POST.get("address") or "").strip()
        emergency_contact = (request.POST.get("emergency_contact") or "").strip()
        serial = (request.POST.get("serial") or "").strip()

        context["form"] = {
            "name": name,
            "age": age_raw,
            "conditions": conditions,
            "address": address,
            "emergency_contact": emergency_contact,
            "serial": serial,
        }

        if not name or not serial:
            context["error"] = "이름과 시리얼 번호는 필수입니다."
            return render(request, "dashboard/device_register.html", context)

        if Device.objects.filter(serial=serial).exists():
            context["error"] = "이미 등록된 시리얼 번호입니다."
            return render(request, "dashboard/device_register.html", context)

        age = None
        if age_raw:
            try:
                age = int(age_raw)
            except ValueError:
                context["error"] = "나이는 숫자로 입력해주세요."
                return render(request, "dashboard/device_register.html", context)
            if age < 0:
                context["error"] = "나이는 0 이상이어야 합니다."
                return render(request, "dashboard/device_register.html", context)

        user_id = generate_user_id()
        baseline = {}
        if age is not None:
            baseline["age"] = age
        if conditions:
            baseline["conditions"] = conditions

        with transaction.atomic():
            owner = Owner.objects.create(
                user_id=user_id,
                name=name,
                age=age,
                conditions=conditions,
                phone="",
            )
            elderly = Elderly.objects.create(
                name=name,
                address=address or "미입력",
                emergency_contact=emergency_contact or "미입력",
                baseline=baseline,
                learning_progress=0,
            )
            is_primary = not Device.objects.filter(owner=owner, is_primary=True).exists()
            device = Device.objects.create(
                owner=owner,
                elderly=elderly,
                serial=serial,
                is_primary=is_primary,
            )

        context["success"] = {
            "user_id": owner.user_id,
            "serial": device.serial,
            "elderly_id": elderly.id,
        }
        context.pop("form", None)

    context["devices"] = Device.objects.select_related("owner", "elderly").order_by("-id")
    return render(request, "dashboard/device_register.html", context)


def device_edit(request, device_id: int):
    device = get_object_or_404(Device.objects.select_related("owner", "elderly"), id=device_id)
    ws_base = os.getenv("WS_BASE", "").strip()
    context = {
        "title": "대상자 정보 수정",
        "page_name": "device_vision",
        "device": device,
            "ws_base": ws_base,
    }

    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        age_raw = (request.POST.get("age") or "").strip()
        conditions = (request.POST.get("conditions") or "").strip()
        address = (request.POST.get("address") or "").strip()
        emergency_contact = (request.POST.get("emergency_contact") or "").strip()
        serial = (request.POST.get("serial") or "").strip()

        if not name or not serial:
            context["error"] = "이름과 시리얼 번호는 필수입니다."
            return render(request, "dashboard/device_edit.html", context)

        if Device.objects.filter(serial=serial).exclude(id=device.id).exists():
            context["error"] = "이미 등록된 시리얼 번호입니다."
            return render(request, "dashboard/device_edit.html", context)

        age = None
        if age_raw:
            try:
                age = int(age_raw)
            except ValueError:
                context["error"] = "나이는 숫자로 입력해주세요."
                return render(request, "dashboard/device_edit.html", context)
            if age < 0:
                context["error"] = "나이는 0 이상이어야 합니다."
                return render(request, "dashboard/device_edit.html", context)

        owner = device.owner
        owner.name = name
        owner.age = age
        owner.conditions = conditions
        owner.save(update_fields=["name", "age", "conditions", "updated_at"])

        elderly = device.elderly
        if elderly is None:
            elderly = Elderly.objects.create(
                name=name,
                address=address or "미입력",
                emergency_contact=emergency_contact or "미입력",
                baseline={},
                learning_progress=0,
            )
            device.elderly = elderly
        elderly.name = name
        elderly.address = address or "미입력"
        elderly.emergency_contact = emergency_contact or "미입력"
        baseline = elderly.baseline or {}
        if age is None:
            baseline.pop("age", None)
        else:
            baseline["age"] = age
        if conditions:
            baseline["conditions"] = conditions
        else:
            baseline.pop("conditions", None)
        elderly.baseline = baseline
        elderly.save(update_fields=["name", "address", "emergency_contact", "baseline"])

        if device.serial != serial:
            device.serial = serial
        device.save(update_fields=["serial", "elderly", "updated_at"])

        context["success"] = "수정이 완료되었습니다."
        context["device"] = device

    return render(request, "dashboard/device_edit.html", context)


def device_vision(request, device_id: int):
    device = get_object_or_404(Device.objects.select_related("owner", "elderly"), id=device_id)
    ws_base = os.getenv("WS_BASE", "").strip()
    return render(
        request,
        "dashboard/device_vision.html",
        {
            "title": "실시간 인식 모니터",
            "page_name": "device_vision",
            "device": device,
            "ws_base": ws_base,
        },
    )
