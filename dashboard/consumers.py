import asyncio
import base64
import json
from threading import Lock
from typing import Any

import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from mediapipe.python.solutions import drawing_utils, face_mesh, pose


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

    async def connect(self):
        self.device_id = self.scope["url_route"]["kwargs"]["device_id"]
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

        async with self._sema:
            try:
                processed_b64 = await asyncio.to_thread(self._process_frame, frame_b64)
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

    @classmethod
    def _process_frame(cls, frame_b64: str) -> str:
        frame_bytes = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            return ""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with cls._mp_lock:
            face_res: Any = cls._face_mesh.process(rgb)
            pose_res: Any = cls._pose.process(rgb)

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
