import json
import re

import os

import tempfile

import threading

import time
import requests
from datetime import timedelta



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



from django.db.models import Count, Q
from django.core.cache import cache
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db.models.functions import TruncDate
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

from .models import (

    AiSetting,

    Alert,

    Conversation,

    Device,

    DeviceToken,

    Elderly,

    ElderlyStatus,

    EmergencyAlert,

    EnvironmentReading,

    HealthLog,

    Quest,

)

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









KEYWORD_TRIGGERS = [

    "\uc544\ud504",

    "\uc678\ub86d",

    "\uc0b4\ub824",

    "\ub3c4\uc640",

    "\ub118\uc5b4",

]





def _generate_reply(transcript: str) -> str:

    message = transcript.strip()

    if not message:

        return "\ub124, \ubcf5\uc774\uc57c \uc5ec\uae30 \uc788\uc5b4\uc694. \uad1c\ucc2e\uc73c\uc138\uc694?"



    low = message.lower()

    if any(token in low for token in ["\uc544\ud30c", "\uc544\ud504", "pain"]):

        return "\uc5b4\ub514\uac00 \uc544\ud504\uc2e0\uac00\uc694? \uc9c0\uae08 \ubc14\ub85c \ub3c4\uc640\ub4dc\ub9b4\uac8c\uc694."

    if any(token in low for token in ["\uc678\ub85c", "lonely", "\uc2ec\uc2ec"]):

        return "\uc81c\uac00 \uac70\uc5d0 \uc788\uc744\uac8c\uc694. \uc624\ub298\uc740 \uc5b4\ub5a4 \uc77c\uc774 \uc788\uc73c\uc168\uc5b4\uc694?"



    return "\ub9d0\uc500 \uc798 \ub4e4\uc5c8\uc5b4\uc694. \uc9c0\uae08 \uae30\ubd84\uc740 \uad1c\ucc2e\uc73c\uc138\uc694?"







LLM_HISTORY_LIMIT = int(os.getenv("LLM_HISTORY_LIMIT", "6"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
LLM_PERSONA_PROMPT = os.getenv(
    "LLM_PERSONA_PROMPT",
    "\ub2e4\uc815\ud55c \uc190\uc8fc \uc785\ub2c8\ub2e4. \uc608\uc758 \ubc14\ub978 \uc5b8\uc5b4\ub85c \uc9e7\uac8c \ub300\ub2f5\ud558\uace0, \ub05d\uc5d0 \ud55c \uac1c \uc9c8\ubb38\uc744 \ud55c\ub2e4.",
)
AI_SETTINGS_CACHE_KEY = "ai_settings_cache"
AI_SETTINGS_CACHE_TTL = int(os.getenv("AI_SETTINGS_CACHE_TTL", "60"))


def _load_ai_settings() -> dict:
    cached = cache.get(AI_SETTINGS_CACHE_KEY)
    if cached is not None:
        return cached

    setting = None
    try:
        setting = AiSetting.objects.order_by("-updated_at").first()
    except Exception:
        setting = None

    data = {
        "persona_prompt": (setting.persona_prompt or "") if setting else "",
        "gemini_api_key": (setting.gemini_api_key or "") if setting else "",
    }
    cache.set(AI_SETTINGS_CACHE_KEY, data, AI_SETTINGS_CACHE_TTL)
    return data


def _resolve_persona_prompt() -> str:
    settings = _load_ai_settings()
    if settings.get("persona_prompt"):
        return settings["persona_prompt"]
    return LLM_PERSONA_PROMPT


def _resolve_gemini_api_key() -> str:
    settings = _load_ai_settings()
    if settings.get("gemini_api_key"):
        return settings["gemini_api_key"]
    return os.getenv("GEMINI_API_KEY", "")


def _build_gemini_contents(history, transcript: str):
    contents = []
    for item in history:
        if item.transcript:
            contents.append({"role": "user", "parts": [{"text": item.transcript}]})
        if item.response:
            contents.append({"role": "model", "parts": [{"text": item.response}]})
    contents.append({"role": "user", "parts": [{"text": transcript}]})
    return contents

def _normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text or "", flags=re.UNICODE).lower()


def _split_wake_words(wake_word: str) -> list[str]:
    if not wake_word:
        return []
    wake_word = wake_word.replace("|", ",")
    return [token.strip() for token in wake_word.split(",") if token.strip()]


def _wake_word_detected(wake_word: str, transcript: str) -> bool:
    normalized_transcript = _normalize_text(transcript)
    if not normalized_transcript:
        return False
    for token in _split_wake_words(wake_word):
        normalized_wake = _normalize_text(token)
        if normalized_wake and normalized_wake in normalized_transcript:
            return True
    return False


def _is_wake_only(wake_word: str, transcript: str) -> bool:
    normalized_transcript = _normalize_text(transcript)
    if not normalized_transcript:
        return False
    for token in _split_wake_words(wake_word):
        normalized_wake = _normalize_text(token)
        if normalized_wake and normalized_transcript == normalized_wake:
            return True
    return False


def _call_gemini(transcript: str, history) -> str:
    api_key = _resolve_gemini_api_key()
    if not api_key:
        return ""

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    payload = {
        "system_instruction": {"parts": [{"text": _resolve_persona_prompt()}]},
        "contents": _build_gemini_contents(history, transcript),
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 256,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    try:
        data = response.json()
    except ValueError:
        return ""

    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return ""
    return str(parts[0].get("text") or "").strip()

ENV_HEAT_THRESHOLD = float(os.getenv("ENV_HEAT_THRESHOLD", "30.0"))
ENV_COLD_THRESHOLD = float(os.getenv("ENV_COLD_THRESHOLD", "15.0"))
ENV_GUIDE_COOLDOWN_SECONDS = int(os.getenv("ENV_GUIDE_COOLDOWN_SECONDS", "3600"))
ENV_PERSONA_PROMPT = os.getenv(
    "ENV_PERSONA_PROMPT",
    "20대 초반 손주 '복이' 페르소나로 예의 바른 살가운 말투로 전라도 사투리를 조건적으로 사용하세요. 1~2문장으로 짧게 말해주고 현재 온도를 반드시 언급해주세요.",
)


def _call_gemini_with_prompt(prompt: str, transcript: str) -> str:
    api_key = _resolve_gemini_api_key()
    if not api_key:
        return ""

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    payload = {
        "system_instruction": {"parts": [{"text": prompt}]},
        "contents": [{"role": "user", "parts": [{"text": transcript}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 128,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    try:
        data = response.json()
    except ValueError:
        return ""

    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return ""
    return str(parts[0].get("text") or "").strip()


def _should_send_environment_message(temperature: float) -> str | None:
    if temperature >= ENV_HEAT_THRESHOLD:
        return "heat"
    if temperature <= ENV_COLD_THRESHOLD:
        return "cold"
    return None


def _environment_cache_key(reason: str) -> str:
    return f"env_last_sent_{reason}"


def _can_send_environment_message(reason: str) -> bool:
    last = cache.get(_environment_cache_key(reason))
    if not last:
        return True
    return (time.time() - float(last)) >= ENV_GUIDE_COOLDOWN_SECONDS


def _mark_environment_sent(reason: str) -> None:
    cache.set(_environment_cache_key(reason), time.time(), ENV_GUIDE_COOLDOWN_SECONDS)


def _generate_environment_message(temperature: float, reason: str) -> str:
    if reason == "heat":
        action = "물을 자주 마시고 시원하게 지내주세요."
    else:
        action = "따뜻하게 옷을 입고 난방 상태를 확인해주세요."

    transcript = (
        f"현재 온도는 {temperature:.1f}도예요. {action}"
        "어르신에게 안내하는 말투로 보내주세요."
    )

    message = _call_gemini_with_prompt(ENV_PERSONA_PROMPT, transcript)
    if message:
        return message

    if reason == "heat":
        return (
            f"어르신, 지금 온도가 {temperature:.1f}도예요."
            " 물 자주 드시고 쉬운 옷으로 시원하게 지내주세요."
        )
    return (
        f"어르신, 현재 온도가 {temperature:.1f}도예요."
        " 따뜻하게 옷 입고 난방 확인해주세요."
    )


def _broadcast_environment_message(message: str, temperature: float, reason: str, announce_only: bool = True) -> None:
    channel_layer = get_channel_layer()
    if channel_layer is None:
        return

    offline_seconds = int(os.getenv("DEVICE_OFFLINE_SECONDS", "300"))
    cutoff = timezone.now() - timedelta(seconds=offline_seconds)
    devices = Device.objects.filter(is_active=True, last_seen_at__gte=cutoff)
    for device in devices:
        log = EnvironmentGuideLog.objects.create(
            device=device,
            temperature=temperature,
            message=message,
            reason=reason,
        )
        async_to_sync(channel_layer.group_send)(
            f"vision_{device.id}",
            {
                "type": "broadcast_tts",
                "message": message,
                "announce_only": announce_only,
                "log_id": log.id,
            },
        )

def _detect_keywords(transcript: str) -> list[str]:

    hits = []

    for keyword in KEYWORD_TRIGGERS:

        if keyword in transcript:

            hits.append(keyword)

    return hits







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









class STTView(APIView):

    parser_classes = [MultiPartParser, FormParser]

    authentication_classes = []

    permission_classes = [AllowAny]



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



        elderly_value = request.data.get("elderly")

        elderly_id = _ensure_elderly(elderly_value)

        device_id = request.data.get("device_id")
        try:
            device_id = int(device_id) if device_id else None
        except (TypeError, ValueError):
            device_id = None
        if device_id:
            Device.objects.filter(id=device_id).update(last_seen_at=timezone.now())



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

        wake_word = str(request.data.get("wake_word") or "").strip()
        wake_detected = _wake_word_detected(wake_word, transcript) if wake_word else True
        wake_only = bool(wake_word and _is_wake_only(wake_word, transcript))

        history = Conversation.objects.filter(elderly_id=elderly_id).order_by("-created_at")[:LLM_HISTORY_LIMIT]
        history = list(reversed(history))

        message = ""
        gemini_response = ""
        if wake_detected and not wake_only:
            gemini_response = _call_gemini(transcript, history)
            message = gemini_response or _generate_reply(transcript)

        keywords = _detect_keywords(transcript)

        if keywords:

            EmergencyAlert.objects.create(

                elderly_id=elderly_id,

                level=EmergencyAlert.LEVEL_WARNING,

                reason="keyword_detected",

                payload={"keywords": keywords, "transcript": transcript},

            )

            Alert.objects.create(

                elderly_id=elderly_id,

                level=Alert.LEVEL_YELLOW,

                risk_type="keyword_detected",

                skeleton_data={"keywords": keywords},

            )



        Conversation.objects.create(

            elderly_id=elderly_id,

            transcript=transcript,

            response=message,

            emotion="",

        )



        return Response(

            {

                "transcript": transcript,

                "message": message,

                "wake_detected": wake_detected,

                "keywords": keywords,

                "language": getattr(info, "language", None),

                "language_probability": getattr(info, "language_probability", None),

            },

            status=status.HTTP_200_OK,

        )







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

        reason = _should_send_environment_message(temperature)
        if reason and _can_send_environment_message(reason):
            def _send():
                message = _generate_environment_message(temperature, reason)
                if message:
                    _broadcast_environment_message(message, temperature, reason, announce_only=True)
                    _mark_environment_sent(reason)

            threading.Thread(target=_send, daemon=True).start()




        return Response({"status": "ok"}, status=status.HTTP_201_CREATED)









class EnvironmentGuideConfirmView(APIView):
    parser_classes = [JSONParser, FormParser]
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        log_id = request.data.get("log_id")
        try:
            log_id = int(log_id)
        except (TypeError, ValueError):
            return Response(
                {"detail": "log_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        log = EnvironmentGuideLog.objects.filter(id=log_id).first()
        if not log:
            return Response(
                {"detail": "Log not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if not log.acknowledged:
            log.acknowledged = True
            log.acknowledged_at = timezone.now()
            log.save(update_fields=["acknowledged", "acknowledged_at"])

        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class AlertResetView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        Alert.objects.all().delete()
        EmergencyAlert.objects.all().delete()
        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class MonitorSummaryView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        latest_alert = EmergencyAlert.objects.select_related("elderly").first()
        latest_convo = Conversation.objects.select_related("elderly").first()
        latest_status = ElderlyStatus.objects.select_related("elderly").first()

        offline_seconds = int(os.getenv("DEVICE_OFFLINE_SECONDS", "300"))
        cutoff = timezone.now() - timedelta(seconds=offline_seconds)
        offline_qs = Device.objects.filter(Q(last_seen_at__lt=cutoff) | Q(last_seen_at__isnull=True))
        offline_devices = [
            {
                "id": device.id,
                "serial": device.serial,
                "last_seen_at": device.last_seen_at.isoformat() if device.last_seen_at else None,
            }
            for device in offline_qs
        ]

        alert_payload = None
        if latest_alert:
            alert_payload = {
                "id": latest_alert.id,
                "level": latest_alert.level,
                "reason": latest_alert.reason,
                "elderly": latest_alert.elderly.name,
                "elderly_id": latest_alert.elderly_id,
                "created_at": latest_alert.created_at.isoformat(),
            }

        convo_payload = None
        if latest_convo:
            convo_payload = {
                "id": latest_convo.id,
                "elderly": latest_convo.elderly.name,
                "elderly_id": latest_convo.elderly_id,
                "transcript": latest_convo.transcript,
                "response": latest_convo.response,
                "created_at": latest_convo.created_at.isoformat(),
            }

        status_payload = None
        if latest_status:
            status_payload = {
                "elderly": latest_status.elderly.name,
                "elderly_id": latest_status.elderly_id,
                "mood_score": latest_status.mood_score,
                "safety_state": latest_status.safety_state,
                "updated_at": latest_status.updated_at.isoformat(),
            }

        return Response(
            {
                "alert": alert_payload,
                "conversation": convo_payload,
                "status": status_payload,
                "offline_devices": offline_devices,
                "offline_count": len(offline_devices),
            },
            status=status.HTTP_200_OK,
        )


class BaselineResetView(APIView):
    parser_classes = [JSONParser, FormParser]
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        elderly_id = request.data.get("elderly_id")
        device_id = request.data.get("device_id")
        if not elderly_id and device_id:
            try:
                device = Device.objects.select_related("elderly").get(id=int(device_id))
                elderly_id = device.elderly_id
            except (Device.DoesNotExist, ValueError, TypeError):
                elderly_id = None

        try:
            elderly_id = int(elderly_id) if elderly_id else None
        except (TypeError, ValueError):
            elderly_id = None

        if not elderly_id:
            return Response({"detail": "elderly_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        from .consumers import VisionConsumer

        VisionConsumer.reset_baseline(elderly_id)
        return Response({"status": "ok", "elderly_id": elderly_id}, status=status.HTTP_200_OK)


class MonitorTrendsView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        days_short = int(os.getenv("TREND_DAYS_SHORT", "7"))
        days_long = int(os.getenv("TREND_DAYS_LONG", "30"))

        def build_series(qs, days):
            end = timezone.now().date()
            start = end - timedelta(days=days - 1)
            base = {start + timedelta(days=i): 0 for i in range(days)}
            data = (
                qs.filter(created_at__date__gte=start)
                .annotate(day=TruncDate("created_at"))
                .values("day")
                .annotate(count=Count("id"))
            )
            for row in data:
                base[row["day"]] = row["count"]
            labels = [d.isoformat() for d in base.keys()]
            values = list(base.values())
            return {"labels": labels, "values": values}

        convo_short = build_series(Conversation.objects.all(), days_short)
        convo_long = build_series(Conversation.objects.all(), days_long)
        emotion_qs = EmergencyAlert.objects.filter(reason="emotion_change")
        emotion_short = build_series(emotion_qs, days_short)
        emotion_long = build_series(emotion_qs, days_long)

        return Response(
            {
                "conversation": {"short": convo_short, "long": convo_long},
                "emotion": {"short": emotion_short, "long": emotion_long},
            },
            status=status.HTTP_200_OK,
        )




class DeviceTTSView(APIView):
    parser_classes = [JSONParser, FormParser]
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        message = str(request.data.get("message") or "").strip()
        device_id = request.data.get("device_id")
        try:
            device_id = int(device_id) if device_id else None
        except (TypeError, ValueError):
            device_id = None

        announce_only_raw = request.data.get("announce_only", False)
        if isinstance(announce_only_raw, str):
            announce_only = announce_only_raw.lower() in {"1", "true", "yes", "y"}
        else:
            announce_only = bool(announce_only_raw)

        if not message or not device_id:
            return Response(
                {"detail": "device_id and message are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        device = Device.objects.select_related("elderly").filter(id=device_id).first()
        if not device:
            return Response(
                {"detail": "Device not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if device.elderly_id and not announce_only:
            Quest.objects.create(
                elderly_id=device.elderly_id,
                title=message,
                is_completed=False,
                badge_image="",
            )

        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"vision_{device_id}",
            {"type": "broadcast_tts", "message": message, "announce_only": announce_only},
        )

        return Response({"status": "ok"}, status=status.HTTP_200_OK)


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

