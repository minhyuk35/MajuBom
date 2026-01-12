import asyncio
import base64
import json
import os
import time
from threading import Lock
from typing import Any, Dict, Optional

import cv2
import numpy as np
from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from django.utils import timezone
import mediapipe as mp

drawing_utils = mp.solutions.drawing_utils
face_mesh = mp.solutions.face_mesh
pose = mp.solutions.pose

from .models import Alert, Device, ElderlyStatus, EmergencyAlert


class VisionConsumer(AsyncWebsocketConsumer):
    _face_mesh = face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _pose = pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _draw = drawing_utils
    _face_connections = list(face_mesh.FACEMESH_TESSELATION)
    _pose_connections = list(pose.POSE_CONNECTIONS)
    _mp_lock = Lock()
    _sema = asyncio.Semaphore(2)

    _state: Dict[int, Dict[str, Any]] = {}
    _device_state: Dict[int, Dict[str, Any]] = {}

    _baseline_samples = int(os.getenv("BASELINE_SAMPLES", "30"))
    _baseline_log_every = max(1, int(os.getenv("BASELINE_LOG_EVERY", "5")))
    _fall_threshold = float(os.getenv("FALL_THRESHOLD", "0.12"))
    _fall_seconds = float(os.getenv("FALL_SECONDS", "1"))
    _emotion_threshold = float(os.getenv("EMOTION_THRESHOLD", "0.03"))
    _emotion_frames = int(os.getenv("EMOTION_FRAMES", "8"))
    _alert_cooldown = float(os.getenv("ALERT_COOLDOWN", "30"))
    _last_seen_interval = float(os.getenv("LAST_SEEN_INTERVAL", "20"))
    _fall_tts_message = os.getenv(
        "FALL_TTS_MESSAGE",
        "\uad1c\ucc2e\uc73c\uc138\uc694? \ub3c4\uc6c0\uc774 \ud544\uc694\ud558\uc2dc\uba74 \ub9d0\uc500\ud574\uc8fc\uc138\uc694.",
    )

    async def connect(self):
        self.device_id = int(self.scope["url_route"]["kwargs"]["device_id"])
        self.elderly_id = await self._fetch_elderly_id(self.device_id)
        self.group_name = f"vision_{self.device_id}"
        self._closed = False
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        self._closed = True
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        if not text_data:
            return
        try:
            payload = json.loads(text_data)
        except json.JSONDecodeError:
            return

        frame_b64 = payload.get("frame")
        if not frame_b64:
            return

        await self._touch_device(self.device_id)

        async with self._sema:
            try:
                processed_b64 = await asyncio.to_thread(
                    self._process_frame,
                    frame_b64,
                    self.elderly_id,
                    self.device_id,
                )
            except Exception:
                return

        if processed_b64:
            await self.channel_layer.group_send(
                self.group_name,
                {"type": "broadcast_frame", "frame": processed_b64},
            )

    async def broadcast_frame(self, event):
        if getattr(self, "_closed", False):
            return
        try:
            await self.send(text_data=json.dumps({"frame": event["frame"]}))
        except Exception:
            return

    async def broadcast_tts(self, event):
        if getattr(self, "_closed", False):
            return
        message = event.get("message")
        if not message:
            return
        announce_only = bool(event.get("announce_only"))
        log_id = event.get("log_id")
        payload = {"tts": message, "announce_only": announce_only}
        if log_id is not None:
            payload["log_id"] = log_id
        try:
            await self.send(text_data=json.dumps(payload))
        except Exception:
            return

    @classmethod
    def _get_state(cls, key: int) -> Dict[str, Any]:
        state = cls._state.get(key)
        if state is None:
            state = {
                "baseline_samples": 0,
                "baseline_pose_sum": 0.0,
                "baseline_brow_sum": 0.0,
                "baseline_mouth_sum": 0.0,
                "baseline_pose": None,
                "baseline_face": None,
                "baseline_log_next": cls._baseline_log_every,
                "baseline_missing_logged": False,
                "fall_start": None,
                "emotion_frames": 0,
                "last_fall_at": 0.0,
                "last_emotion_at": 0.0,
            }
            cls._state[key] = state
        return state


    @classmethod
    def _get_device_state(cls, device_id: int) -> Dict[str, Any]:
        state = cls._device_state.get(device_id)
        if state is None:
            state = {
                "last_seen_at": 0.0,
            }
            cls._device_state[device_id] = state
        return state



    @classmethod
    def reset_baseline(cls, elderly_id: int) -> None:
        state = cls._state.get(elderly_id)
        if state is None:
            state = cls._get_state(elderly_id)
        state.update(
            {
                "baseline_samples": 0,
                "baseline_pose_sum": 0.0,
                "baseline_brow_sum": 0.0,
                "baseline_mouth_sum": 0.0,
                "baseline_pose": None,
                "baseline_face": None,
                "baseline_log_next": cls._baseline_log_every,
                "baseline_missing_logged": False,
                "fall_start": None,
                "emotion_frames": 0,
                "last_fall_at": 0.0,
                "last_emotion_at": 0.0,
            }
        )
        try:
            status, _ = ElderlyStatus.objects.get_or_create(elderly_id=elderly_id)
            status.baseline_pose = {}
            status.baseline_face = {}
            status.safety_state = "normal"
            status.save(update_fields=["baseline_pose", "baseline_face", "safety_state", "updated_at"])
        except Exception:
            return


    @classmethod
    def _process_frame(
        cls,
        frame_b64: str,
        elderly_id: Optional[int],
        device_id: Optional[int],
    ) -> str:
        frame_bytes = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            return ""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with cls._mp_lock:
            face_res: Any = cls._face_mesh.process(rgb)
            pose_res: Any = cls._pose.process(rgb)

        hip_y = cls._extract_hip_y(pose_res)
        face_metrics = cls._extract_face_metrics(face_res)
        skeleton_points = cls._extract_skeleton_points(pose_res)

        if elderly_id:
            cls._update_baseline(elderly_id, hip_y, face_metrics)
            cls._detect_fall(elderly_id, hip_y, device_id, skeleton_points)
            cls._detect_emotion(elderly_id, face_metrics)

        if face_res.multi_face_landmarks:
            for face_landmarks in face_res.multi_face_landmarks:
                cls._draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    cls._face_connections,
                )

        if pose_res.pose_landmarks:
            cls._draw.draw_landmarks(
                frame,
                pose_res.pose_landmarks,
                cls._pose_connections,
            )

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return ""

        return base64.b64encode(buf).decode("ascii")

    @classmethod
    def _extract_hip_y(cls, pose_res: Any) -> Optional[float]:
        if not pose_res.pose_landmarks:
            return None
        landmarks = pose_res.pose_landmarks.landmark
        left = landmarks[23].y
        right = landmarks[24].y
        return float((left + right) / 2.0)

    @classmethod
    def _extract_face_metrics(cls, face_res: Any) -> Optional[Dict[str, float]]:
        if not face_res.multi_face_landmarks:
            return None
        landmarks = face_res.multi_face_landmarks[0].landmark
        brow_left = abs(landmarks[65].y - landmarks[159].y)
        brow_right = abs(landmarks[295].y - landmarks[386].y)
        mouth_open = abs(landmarks[13].y - landmarks[14].y)
        return {
            "brow": float((brow_left + brow_right) / 2.0),
            "mouth": float(mouth_open),
        }

    @classmethod
    def _extract_skeleton_points(cls, pose_res: Any) -> Optional[list[Dict[str, float]]]:
        if not pose_res.pose_landmarks:
            return None
        landmarks = pose_res.pose_landmarks.landmark

        def _clamp(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        def _pt(x: float, y: float) -> Dict[str, float]:
            return {"x": round(_clamp(x) * 100, 1), "y": round(_clamp(y) * 100, 1)}

        def _mid(a, b) -> tuple[float, float]:
            return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

        head = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        left_ankle = landmarks[27]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]
        right_foot = landmarks[32]

        chest_x, chest_y = _mid(left_shoulder, right_shoulder)
        hip_x, hip_y = _mid(left_hip, right_hip)

        return [
            _pt(head.x, head.y),
            _pt(chest_x, chest_y),
            _pt(left_shoulder.x, left_shoulder.y),
            _pt(left_elbow.x, left_elbow.y),
            _pt(right_shoulder.x, right_shoulder.y),
            _pt(right_elbow.x, right_elbow.y),
            _pt(right_wrist.x, right_wrist.y),
            _pt(hip_x, hip_y),
            _pt(left_knee.x, left_knee.y),
            _pt(left_ankle.x, left_ankle.y),
            _pt(right_knee.x, right_knee.y),
            _pt(right_ankle.x, right_ankle.y),
            _pt(right_foot.x, right_foot.y),
        ]

    @classmethod
    def _update_baseline(
        cls,
        elderly_id: int,
        hip_y: Optional[float],
        face_metrics: Optional[Dict[str, float]],
    ) -> None:
        state = cls._get_state(elderly_id)
        if state["baseline_pose"] is not None and state["baseline_face"] is not None:
            return
        if hip_y is None or face_metrics is None:
            return

        state["baseline_samples"] += 1
        state["baseline_pose_sum"] += hip_y
        state["baseline_brow_sum"] += face_metrics["brow"]
        state["baseline_mouth_sum"] += face_metrics["mouth"]

        next_log = state.get("baseline_log_next") or cls._baseline_log_every
        if state["baseline_samples"] >= next_log and state["baseline_samples"] < cls._baseline_samples:
            print(
                f"[BASELINE] samples={state['baseline_samples']}/{cls._baseline_samples} "
                f"hip_y={hip_y:.3f}"
            )
            state["baseline_log_next"] = next_log + cls._baseline_log_every

        if state["baseline_samples"] < cls._baseline_samples:
            return

        baseline_pose = state["baseline_pose_sum"] / max(state["baseline_samples"], 1)
        baseline_face = {
            "brow": state["baseline_brow_sum"] / max(state["baseline_samples"], 1),
            "mouth": state["baseline_mouth_sum"] / max(state["baseline_samples"], 1),
        }
        state["baseline_pose"] = baseline_pose
        state["baseline_face"] = baseline_face
        state["baseline_missing_logged"] = False
        print(f"[BASELINE] ready hip_y={baseline_pose:.3f}")

        try:
            status, _ = ElderlyStatus.objects.get_or_create(elderly_id=elderly_id)
            status.baseline_pose = {"hip_y": baseline_pose}
            status.baseline_face = baseline_face
            status.save(update_fields=["baseline_pose", "baseline_face", "updated_at"])
        except Exception:
            return

    @classmethod
    def _detect_fall(
        cls,
        elderly_id: int,
        hip_y: Optional[float],
        device_id: Optional[int],
        skeleton_points: Optional[list[Dict[str, float]]],
    ) -> None:
        if hip_y is None:
            return
        state = cls._get_state(elderly_id)
        baseline = state.get("baseline_pose")
        if baseline is None:
            if not state.get("baseline_missing_logged"):
                print(f"[FALL] baseline not ready (samples={state['baseline_samples']})")
                state["baseline_missing_logged"] = True
            return

        delta = hip_y - float(baseline)
        if delta >= cls._fall_threshold:
            if state.get("fall_start") is None:
                state["fall_start"] = time.time()
                print(
                    f"[FALL] candidate start delta={delta:.3f} "
                    f"threshold={cls._fall_threshold:.3f}"
                )
        else:
            if state.get("fall_start") is not None:
                print(
                    f"[FALL] candidate cleared delta={delta:.3f} "
                    f"threshold={cls._fall_threshold:.3f}"
                )
            state["fall_start"] = None
            return

        fall_start = state.get("fall_start")
        if not fall_start or (time.time() - fall_start) < cls._fall_seconds:
            return

        now = time.time()
        if now - state["last_fall_at"] < cls._alert_cooldown:
            return

        state["last_fall_at"] = now
        state["fall_start"] = None
        print(
            f"[FALL] confirmed delta={delta:.3f} "
            f"duration>={cls._fall_seconds:.1f}s"
        )

        try:
            skeleton_data = {
                "detail": "Fall detected",
                "note": "Pose threshold exceeded",
                "delta": delta,
            }
            if skeleton_points:
                skeleton_data["points"] = skeleton_points

            EmergencyAlert.objects.create(
                elderly_id=elderly_id,
                level=EmergencyAlert.LEVEL_CRITICAL,
                reason="fall_detected",
                payload={"delta": delta},
            )
            Alert.objects.create(
                elderly_id=elderly_id,
                level=Alert.LEVEL_RED,
                risk_type="fall_detected",
                skeleton_data=skeleton_data,
            )
            status, _ = ElderlyStatus.objects.get_or_create(elderly_id=elderly_id)
            status.safety_state = "fall"
            status.save(update_fields=["safety_state", "updated_at"])
            cls._broadcast_tts(device_id, cls._fall_tts_message)
        except Exception:
            return


    async def _touch_device(self, device_id: int) -> None:
        state = self._get_device_state(device_id)
        now = time.time()
        if (now - state["last_seen_at"]) < self._last_seen_interval:
            return
        state["last_seen_at"] = now
        await self._update_last_seen(device_id)

    @staticmethod
    @database_sync_to_async
    def _update_last_seen(device_id: int) -> None:
        Device.objects.filter(id=device_id).update(last_seen_at=timezone.now())


    @classmethod
    def _detect_emotion(
        cls,
        elderly_id: int,
        face_metrics: Optional[Dict[str, float]],
    ) -> None:
        if face_metrics is None:
            return
        state = cls._get_state(elderly_id)
        baseline = state.get("baseline_face")
        if not baseline:
            return

        brow_delta = baseline["brow"] - face_metrics["brow"]
        mouth_delta = face_metrics["mouth"] - baseline["mouth"]
        score = max(0.0, brow_delta) + max(0.0, mouth_delta)

        if score >= cls._emotion_threshold:
            state["emotion_frames"] += 1
        else:
            state["emotion_frames"] = 0

        if state["emotion_frames"] < cls._emotion_frames:
            return

        now = time.time()
        if now - state["last_emotion_at"] < cls._alert_cooldown:
            return

        state["last_emotion_at"] = now
        state["emotion_frames"] = 0

        try:
            EmergencyAlert.objects.create(
                elderly_id=elderly_id,
                level=EmergencyAlert.LEVEL_WARNING,
                reason="emotion_change",
                payload={"score": score},
            )
            Alert.objects.create(
                elderly_id=elderly_id,
                level=Alert.LEVEL_YELLOW,
                risk_type="emotion_change",
                skeleton_data={"score": score},
            )
            status, _ = ElderlyStatus.objects.get_or_create(elderly_id=elderly_id)
            status.mood_score = score
            status.save(update_fields=["mood_score", "updated_at"])
        except Exception:
            return

    @staticmethod
    def _broadcast_tts(device_id: Optional[int], message: str, announce_only: bool = False) -> None:
        if not device_id or not message:
            return
        channel_layer = get_channel_layer()
        if channel_layer is None:
            return
        async_to_sync(channel_layer.group_send)(
            f"vision_{device_id}",
            {
                "type": "broadcast_tts",
                "message": message,
                "announce_only": announce_only,
            },
        )

    @staticmethod
    @database_sync_to_async
    def _fetch_elderly_id(device_id: int) -> Optional[int]:
        device = Device.objects.select_related("elderly").filter(id=device_id).first()
        if not device:
            return None
        return device.elderly_id

