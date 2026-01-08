import json
import os
import tempfile
import threading
import time

_VISION_IMPORT_ERROR = None
try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency
    cv2 = None
    mp = None
    np = None
    _VISION_IMPORT_ERROR = exc
else:
    if not hasattr(mp, "solutions"):
        cv2 = None
        np = None
        _VISION_IMPORT_ERROR = RuntimeError(
            "mediapipe.solutions is not available; install a mediapipe build that includes solutions."
        )

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - runtime dependency
    WhisperModel = None

from django.http import HttpResponse
from django.utils import timezone
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .device_auth import (
    DeviceAuthRequired,
    DeviceTokenAuthentication,
    generate_token,
    hash_token,
)
from .models import Device, DeviceToken, Elderly, EnvironmentReading, HealthLog, Quest
from . import vision_store

_VISION_LOCK = threading.Lock()
_VISION_STATE = {}
_FACE_MESH = None
_POSE = None

_WHISPER_LOCK = threading.Lock()
_WHISPER_MODEL = None


def _average(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _coerce_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ensure_elderly(elderly_value):
    elderly_id = _coerce_int(elderly_value)
    if not elderly_id:
        default_id = _coerce_int(os.getenv("DEFAULT_ELDERLY_ID"))
        elderly_id = default_id or 1

    Elderly.objects.get_or_create(
        id=elderly_id,
        defaults={
            "name": f"Unregistered {elderly_id}",
            "address": "Unknown",
            "emergency_contact": "Unknown",
            "baseline": {},
            "learning_progress": 0,
        },
    )
    return elderly_id


def _parse_environment(data):
    environment = data.get("environment")
    if isinstance(environment, str):
        try:
            environment = json.loads(environment)
        except json.JSONDecodeError:
            environment = {}
    if not isinstance(environment, dict):
        environment = {}

    temperature = data.get("temperature")
    humidity = data.get("humidity")
    if temperature is not None and "temperature" not in environment:
        environment["temperature"] = temperature
    if humidity is not None and "humidity" not in environment:
        environment["humidity"] = humidity
    return environment


def _get_vision_models():
    global _FACE_MESH, _POSE
    if _FACE_MESH is None or _POSE is None:
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        _POSE = mp.solutions.pose.Pose(model_complexity=1)
    return _FACE_MESH, _POSE


def _annotate_frame(frame, face_result, pose_result):
    annotated = frame.copy()
    drawing_utils = mp.solutions.drawing_utils
    face_mesh_module = mp.solutions.face_mesh
    pose_module = mp.solutions.pose

    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            try:
                drawing_utils.draw_landmarks(
                    annotated,
                    face_landmarks,
                    face_mesh_module.FACEMESH_TESSELATION,
                )
            except Exception:
                continue

    if pose_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            annotated,
            pose_result.pose_landmarks,
            pose_module.POSE_CONNECTIONS,
        )

    return annotated


def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        model_name = os.getenv("AUDIO_MODEL", "base")
        device = os.getenv("AUDIO_DEVICE", "cpu")
        compute_type = os.getenv("AUDIO_COMPUTE_TYPE", "int8")
        _WHISPER_MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _WHISPER_MODEL


def process_vision_frame(frame_bytes, elderly_id, device_id=None, environment=None):
    if cv2 is None or mp is None or np is None:
        detail = "Vision dependencies are not installed."
        if _VISION_IMPORT_ERROR:
            detail = f"Vision dependencies unavailable: {_VISION_IMPORT_ERROR}"
        raise RuntimeError(detail)

    elderly_id = _ensure_elderly(elderly_id)
    frame_array = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image payload.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with _VISION_LOCK:
        face_mesh, pose = _get_vision_models()
        face_result = face_mesh.process(rgb)
        pose_result = pose.process(rgb)

    face_detected = 1.0 if face_result.multi_face_landmarks else 0.0
    activity_level = 0.0
    points = None
    if pose_result.pose_landmarks:
        points = np.array(
            [(lm.x, lm.y) for lm in pose_result.pose_landmarks.landmark],
            dtype=np.float32,
        )

    state = _VISION_STATE.setdefault(
        elderly_id,
        {
            "face_samples": [],
            "activity_samples": [],
            "prev_pose": None,
            "last_post": time.time(),
        },
    )

    if points is not None:
        prev_pose = state.get("prev_pose")
        if prev_pose is not None and prev_pose.shape == points.shape:
            diffs = np.linalg.norm(points - prev_pose, axis=1)
            activity_level = float(np.mean(diffs) * 100.0)
        state["prev_pose"] = points

    state["face_samples"].append(face_detected)
    state["activity_samples"].append(activity_level)

    annotated = _annotate_frame(frame, face_result, pose_result)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok and device_id is not None:
        vision_store.set_latest(
            device_id,
            encoded.tobytes(),
            face_detected,
            activity_level,
        )

    interval = float(os.getenv("VISION_POST_INTERVAL", "60"))
    now = time.time()
    posted = False
    avg_face = None
    avg_activity = None
    if interval <= 0 or (now - state["last_post"]) >= interval:
        avg_face = _average(state["face_samples"])
        avg_activity = _average(state["activity_samples"])
        HealthLog.objects.create(
            elderly_id=elderly_id,
            face_score=avg_face,
            activity_level=avg_activity,
            environment=environment or {},
        )
        state["face_samples"].clear()
        state["activity_samples"].clear()
        state["last_post"] = now
        posted = True

    return {
        "status": "ok",
        "posted": posted,
        "face_score": face_detected,
        "activity_level": activity_level,
        "avg_face_score": avg_face,
        "avg_activity_level": avg_activity,
        "samples": len(state["face_samples"]),
    }


class VisionFrameView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    authentication_classes = [DeviceTokenAuthentication]
    permission_classes = [DeviceAuthRequired]

    def post(self, request, *args, **kwargs):
        frame_file = request.FILES.get("frame") or request.FILES.get("image")
        if not frame_file:
            return Response(
                {"detail": "Missing frame file."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        elderly_value = request.data.get("elderly") or getattr(request.device, "elderly_id", None)
        frame_bytes = frame_file.read()
        if not frame_bytes:
            return Response(
                {"detail": "Empty frame payload."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        environment = _parse_environment(request.data)
        device_id = getattr(request, "device", None)
        if device_id is not None:
            device_id = request.device.id

        try:
            result = process_vision_frame(
                frame_bytes,
                elderly_value,
                device_id=device_id,
                environment=environment,
            )
        except ValueError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except RuntimeError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(result, status=status.HTTP_200_OK)


class AudioClipView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    authentication_classes = [DeviceTokenAuthentication]
    permission_classes = [DeviceAuthRequired]

    def post(self, request, *args, **kwargs):
        if WhisperModel is None:
            return Response(
                {"detail": "Audio dependencies are not installed."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        audio_file = request.FILES.get("audio") or request.FILES.get("file")
        if not audio_file:
            return Response(
                {"detail": "Missing audio file."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        elderly_value = request.data.get("elderly") or getattr(request.device, "elderly_id", None)
        elderly_id = _ensure_elderly(elderly_value)
        suffix = os.path.splitext(audio_file.name or "")[1] or ".wav"

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in audio_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name

            language = os.getenv("AUDIO_LANGUAGE", "ko")
            if language.lower() == "auto":
                language = None
            beam_size = int(os.getenv("AUDIO_BEAM_SIZE", "5"))
            try:
                with _WHISPER_LOCK:
                    model = _get_whisper_model()
                    segments, info = model.transcribe(
                        temp_path,
                        language=language,
                        beam_size=beam_size,
                    )

                transcript_parts = [
                    segment.text.strip() for segment in segments if segment.text.strip()
                ]
                transcript = " ".join(transcript_parts)
            except Exception:
                return Response(
                    {"detail": "Transcription failed."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        max_chars = int(os.getenv("AUDIO_MAX_MESSAGE_CHARS", "200"))
        message = transcript.strip()
        if message:
            message = message[:max_chars]

        if os.getenv("AUDIO_CREATE_QUEST", "0") == "1" and message:
            Quest.objects.create(
                elderly_id=elderly_id,
                title=message,
                is_completed=False,
                badge_image="",
            )

        return Response(
            {
                "transcript": transcript,
                "message": message,
                "language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
            },
            status=status.HTTP_200_OK,
        )


class DeviceHandshakeView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serial = str(request.data.get("serial") or "").strip()
        if not serial:
            return Response(
                {"detail": "serial is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            device = Device.objects.select_related("owner", "elderly").get(
                serial=serial,
                is_active=True,
            )
        except Device.DoesNotExist:
            return Response(
                {"detail": "Device not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if os.getenv("REVOKE_OLD_DEVICE_TOKENS", "1") == "1":
            DeviceToken.objects.filter(device=device, revoked=False).update(revoked=True)

        raw_token = generate_token()
        DeviceToken.objects.create(device=device, token_hash=hash_token(raw_token))
        device.last_seen_at = timezone.now()
        device.save(update_fields=["last_seen_at"])

        config = device.config if isinstance(device.config, dict) else {}
        config = {
            "detection_mode": device.detection_mode,
            "alert_enabled": device.alert_enabled,
            **config,
        }

        return Response(
            {
                "token": raw_token,
                "device_id": device.id,
                "is_primary": device.is_primary,
                "elderly_id": device.elderly_id,
                "config": config,
            },
            status=status.HTTP_200_OK,
        )




class EnvironmentReadingView(APIView):
    parser_classes = [JSONParser, FormParser]
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            temperature = float(request.data.get("temperature"))
            humidity = float(request.data.get("humidity"))
        except (TypeError, ValueError):
            return Response(
                {"detail": "temperature and humidity are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        source = str(request.data.get("source") or "")
        EnvironmentReading.objects.create(
            temperature=temperature,
            humidity=humidity,
            source=source[:64],
        )

        return Response({"status": "ok"}, status=status.HTTP_201_CREATED)


class DeviceVisionImageView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request, device_id: int):
        payload = vision_store.get_latest(device_id)
        if not payload or not payload.get("image"):
            return Response(
                {"detail": "No frame available yet."},
                status=status.HTTP_404_NOT_FOUND,
            )

        response = HttpResponse(payload["image"], content_type="image/jpeg")
        response["Cache-Control"] = "no-store"
        return response
