from __future__ import annotations

from django.core.management.base import BaseCommand

from dashboard.models import Alert, Elderly, HealthLog, Quest


class Command(BaseCommand):
    help = "Seed demo data for SmartHome dashboard"

    def handle(self, *args, **options):
        if Elderly.objects.exists():
            self.stdout.write("Elderly data already exists. Skipping seed.")
            return

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

        elders = [
            {
                "name": "김순자",
                "address": "전주시 노송동 12-3",
                "emergency_contact": "아들 김OO 010-1234-5678",
                "baseline": {
                    "age": 79,
                    "conditions": "고혈압, 관절염",
                    "current_day": 5,
                    "total_days": 7,
                    "metrics": [
                        {"label": "심박수 베이스라인", "value": "72 bpm"},
                        {"label": "보행 안정성", "value": "92%"},
                        {"label": "수면 패턴 일치도", "value": "88%"},
                    ],
                    "sleep": {"total": 6.5, "deep": 2.1, "disruptions": 3},
                },
                "learning_progress": 71,
                "alert": {
                    "level": Alert.LEVEL_RED,
                    "risk_type": "낙상 감지",
                    "handled": True,
                    "skeleton_data": {
                        "detail": "거실에서 급격한 자세 변화",
                        "note": "움직임 불균형",
                        "points": base_points,
                    },
                },
                "environment": {"temperature": 34, "humidity": 78},
                "quests": [
                    {"title": "거실 5분 걷기", "is_completed": True, "badge_image": "badge_walk.png"},
                    {"title": "옛날 전주역 이름 맞추기", "is_completed": False, "badge_image": "badge_memory.png"},
                ],
            },
            {
                "name": "박말순",
                "address": "전주시 완산동 88-2",
                "emergency_contact": "딸 박OO 010-8899-2200",
                "baseline": {
                    "age": 84,
                    "conditions": "당뇨, 경도 치매",
                    "current_day": 7,
                    "total_days": 7,
                    "metrics": [
                        {"label": "심박수 베이스라인", "value": "68 bpm"},
                        {"label": "보행 안정성", "value": "86%"},
                        {"label": "수면 패턴 일치도", "value": "91%"},
                    ],
                    "sleep": {"total": 7.2, "deep": 2.7, "disruptions": 2},
                },
                "learning_progress": 100,
                "alert": {
                    "level": Alert.LEVEL_YELLOW,
                    "risk_type": "장시간 무반응",
                    "handled": False,
                    "skeleton_data": {
                        "detail": "침실 활동 2시간 정지",
                        "note": "호흡 약화",
                        "points": base_points,
                    },
                },
                "environment": {"temperature": 29, "humidity": 60},
                "quests": [
                    {"title": "오늘의 날씨 설명하기", "is_completed": False, "badge_image": "badge_voice.png"},
                ],
            },
            {
                "name": "이기철",
                "address": "전주시 덕진동 402-1",
                "emergency_contact": "아내 이OO 010-5533-1122",
                "baseline": {
                    "age": 81,
                    "conditions": "고지혈증",
                    "current_day": 3,
                    "total_days": 7,
                    "metrics": [
                        {"label": "심박수 베이스라인", "value": "75 bpm"},
                        {"label": "보행 안정성", "value": "89%"},
                        {"label": "수면 패턴 일치도", "value": "80%"},
                    ],
                    "sleep": {"total": 5.9, "deep": 1.8, "disruptions": 4},
                },
                "learning_progress": 43,
                "alert": {
                    "level": Alert.LEVEL_YELLOW,
                    "risk_type": "표정 악화",
                    "handled": False,
                    "skeleton_data": {
                        "detail": "안면 근육 긴장",
                        "note": "표정 경직",
                        "points": base_points,
                    },
                },
                "environment": {"temperature": 26, "humidity": 55},
                "quests": [
                    {"title": "내가 좋아하는 노래 한 소절", "is_completed": True, "badge_image": "badge_music.png"},
                ],
            },
        ]

        for entry in elders:
            elder = Elderly.objects.create(
                name=entry["name"],
                address=entry["address"],
                emergency_contact=entry["emergency_contact"],
                baseline=entry["baseline"],
                learning_progress=entry["learning_progress"],
            )

            alert = entry.get("alert")
            if alert:
                Alert.objects.create(
                    elderly=elder,
                    level=alert["level"],
                    risk_type=alert["risk_type"],
                    handled=alert["handled"],
                    skeleton_data=alert["skeleton_data"],
                )

            HealthLog.objects.create(
                elderly=elder,
                face_score=0.8,
                activity_level=2.1,
                environment=entry["environment"],
            )

            for quest in entry.get("quests", []):
                Quest.objects.create(
                    elderly=elder,
                    title=quest["title"],
                    is_completed=quest["is_completed"],
                    badge_image=quest["badge_image"],
                )

        Quest.objects.create(
            elderly=None,
            title="옛날 전주역 이름 맞추기",
            is_completed=False,
            badge_image="badge_memory.png",
        )

        self.stdout.write(self.style.SUCCESS("Seeded demo data."))
