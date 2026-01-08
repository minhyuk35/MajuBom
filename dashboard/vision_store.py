import threading
from datetime import datetime
from typing import Dict, Optional

_LOCK = threading.Lock()
_LATEST: Dict[int, Dict[str, object]] = {}


def set_latest(device_id: int, image: bytes, face_detected: float, activity_level: float) -> None:
    with _LOCK:
        _LATEST[device_id] = {
            "image": image,
            "updated_at": datetime.utcnow(),
            "face_detected": face_detected,
            "activity_level": activity_level,
        }


def get_latest(device_id: int) -> Optional[Dict[str, object]]:
    with _LOCK:
        return _LATEST.get(device_id)
