import asyncio
import base64
import json
import os
import cv2
import websockets

WS_URL = os.getenv("WS_URL", "ws://192.168.0.30:8000/ws/vision/1/")
FPS = float(os.getenv("FPS", "12"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
WIDTH = int(os.getenv("FRAME_WIDTH", "480"))
HEIGHT = int(os.getenv("FRAME_HEIGHT", "360"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))


async def stream() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Camera not available")
        return

    frame_interval = 1.0 / FPS if FPS > 0 else 0.0

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        await asyncio.sleep(0.01)
                        continue

                    frame = cv2.resize(frame, (WIDTH, HEIGHT))
                    ok, buf = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    if not ok:
                        continue

                    payload = {"frame": base64.b64encode(buf).decode("ascii")}
                    await ws.send(json.dumps(payload))

                    if frame_interval:
                        await asyncio.sleep(frame_interval)
        except Exception as exc:
            print(f"[WARN] WS reconnect: {exc}")
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(stream())
