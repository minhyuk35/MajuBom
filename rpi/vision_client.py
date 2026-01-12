import asyncio
import audioop
import base64
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import wave
import tkinter as tk

import cv2
import requests
import websockets

try:
    from gtts import gTTS
except Exception:
    gTTS = None

WS_URL_OVERRIDE = os.getenv("WS_URL", "").strip()
WS_URL_TEMPLATE = os.getenv("WS_URL_TEMPLATE", "ws://192.168.0.30:8000/ws/vision/{device_id}/")
STT_ENDPOINT = os.getenv("STT_ENDPOINT", "http://192.168.0.30:8000/api/stt/")
ENV_CONFIRM_ENDPOINT = os.getenv("ENV_CONFIRM_ENDPOINT", "http://192.168.0.30:8000/api/environment/confirm/")
HANDSHAKE_ENDPOINT = os.getenv("HANDSHAKE_ENDPOINT", "http://192.168.0.30:8000/api/device-handshake/")
DEVICE_ID = os.getenv("DEVICE_ID", "1")
DEVICE_SERIAL = os.getenv("DEVICE_SERIAL", "").strip()

FPS = float(os.getenv("FPS", "12"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
WIDTH = int(os.getenv("FRAME_WIDTH", "480"))
HEIGHT = int(os.getenv("FRAME_HEIGHT", "360"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "1") == "1"
AUDIO_DEVICE = os.getenv("ARECORD_DEVICE", "plughw:3,0")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHUNK_SEC = float(os.getenv("AUDIO_CHUNK_SEC", "0.5"))
SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "0.6"))
AUDIO_RMS_THRESHOLD = int(os.getenv("AUDIO_RMS_THRESHOLD", "500"))
WAKE_WORD = os.getenv("WAKE_WORD", "\ubcf4\uae30\uc57c,\ubcf5\uc774\uc57c")
WAKE_ACTIVE_SECONDS = float(os.getenv("WAKE_ACTIVE_SECONDS", "30"))
WAKE_PENDING_SECONDS = float(os.getenv("WAKE_PENDING_SECONDS", "4"))
WAKE_RESPONSE = os.getenv("WAKE_RESPONSE", "\ub124, \ub9d0\uc500\ud574\uc8fc\uc138\uc694.")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
STT_RETRIES = int(os.getenv("STT_RETRIES", "3"))
STT_RETRY_BACKOFF = float(os.getenv("STT_RETRY_BACKOFF", "1.5"))

TTS_ENABLED = os.getenv("TTS_ENABLED", "1") == "1"
TTS_LANG = os.getenv("TTS_LANG", "ko")
TTS_RATE = int(os.getenv("TTS_RATE", "160"))
TTS_ENGINE = os.getenv("TTS_ENGINE", "gtts")

_wake_until = 0.0
_wake_lock = threading.Lock()
_wake_pending_until = 0.0
_wake_pending_lock = threading.Lock()
TTS_NOTIFICATION = os.getenv(
    "TTS_NOTIFICATION",
    "\uba54\uc2dc\uc9c0\uac00 \ub3c4\ucc29\ud588\uc5b4\uc694. \ud655\uc778\ud574\uc8fc\uc138\uc694.",
)


class DisplayManager:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.configure(bg="#f5f0e6")
        self.root.withdraw()

        self.message_var = tk.StringVar(value="")
        self._queue: "queue.Queue[tuple[str, int | None]]" = queue.Queue()
        self._last_log_id = None
        self._session = requests.Session()
        self._build_ui()

        self.root.after(100, self._poll_queue)

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg="#f5f0e6")
        container.pack(fill="both", expand=True)

        wrap_length = int(self.root.winfo_screenwidth() * 0.9)
        label = tk.Label(
            container,
            textvariable=self.message_var,
            font=("Helvetica", 52, "bold"),
            fg="#111111",
            bg="#f5f0e6",
            wraplength=wrap_length,
            justify="center",
        )
        label.pack(expand=True, padx=60, pady=40)

        button = tk.Button(
            container,
            text="\ud655\uc778\ud588\uc2b5\ub2c8\ub2e4",
            font=("Helvetica", 32, "bold"),
            fg="#111111",
            bg="#f28c28",
            activebackground="#f7a347",
            activeforeground="#111111",
            command=self.hide,
            padx=40,
            pady=20,
            relief="flat",
        )
        button.pack(pady=40)

    def post_message(self, message: str, log_id: int | None = None) -> None:
        self._queue.put((message, log_id))

    def _poll_queue(self) -> None:
        try:
            while True:
                message, log_id = self._queue.get_nowait()
                self._show(message, log_id)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _show(self, message: str, log_id: int | None) -> None:
        self.message_var.set(message)
        self._last_log_id = log_id
        self.root.deiconify()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.focus_force()

    def _send_ack(self) -> None:
        if not self._last_log_id:
            return
        if not ENV_CONFIRM_ENDPOINT:
            return
        payload = {"log_id": self._last_log_id}
        if DEVICE_ID:
            payload["device_id"] = DEVICE_ID
        try:
            self._session.post(ENV_CONFIRM_ENDPOINT, json=payload, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            print(f"[WARN] Confirm send failed: {exc}")
        self._last_log_id = None

    def hide(self) -> None:
        self._send_ack()
        self.root.attributes("-fullscreen", False)
        self.root.withdraw()

    def run(self) -> None:
        self.root.mainloop()

def speak(text: str) -> None:
    if not TTS_ENABLED:
        return

    message = text.strip()
    if not message:
        return

    if TTS_ENGINE == "gtts" and gTTS is not None:
        tmp_path = None
        try:
            tts = gTTS(message, lang=TTS_LANG)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            tts.save(tmp_path)

            if shutil.which("mpg123"):
                subprocess.run(
                    ["mpg123", "-q", tmp_path],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            if shutil.which("ffplay"):
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", tmp_path],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return

            print("[WARN] No MP3 player found for gTTS output.")
            return
        except Exception as exc:
            print(f"[WARN] gTTS failed: {exc}")
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if shutil.which("espeak-ng"):
        command = ["espeak-ng", "-v", TTS_LANG, "-s", str(TTS_RATE), message]
    elif shutil.which("espeak"):
        command = ["espeak", "-v", TTS_LANG, "-s", str(TTS_RATE), message]
    else:
        print(f"[TTS] {message}")
        return

    subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _resolve_ws_url(device_id: str) -> str:
    if WS_URL_OVERRIDE:
        return WS_URL_OVERRIDE
    return WS_URL_TEMPLATE.format(device_id=device_id)

def _wake_words() -> list[str]:
    if not WAKE_WORD:
        return []
    raw = WAKE_WORD.replace("|", ",")
    return [token.strip() for token in raw.split(",") if token.strip()]

def _normalize_text(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        elif not ch.isspace():
            continue
    return "".join(cleaned)

def _is_wake_only(transcript: str) -> bool:
    normalized_transcript = _normalize_text(transcript.lower())
    if not normalized_transcript:
        return False
    for wake_word in _wake_words():
        normalized_wake = _normalize_text(wake_word.lower())
        if normalized_wake and normalized_transcript == normalized_wake:
            return True
    return False


def _handshake_device() -> str:
    if not DEVICE_SERIAL:
        return DEVICE_ID

    payload = {"serial": DEVICE_SERIAL}
    try:
        response = requests.post(HANDSHAKE_ENDPOINT, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] Device handshake failed: {exc}")
        return DEVICE_ID

    try:
        data = response.json()
    except ValueError:
        print("[WARN] Device handshake returned invalid JSON.")
        return DEVICE_ID

    device_id = data.get("device_id")
    if device_id:
        return str(device_id)
    return DEVICE_ID




def _is_wake_pending() -> bool:
    if not WAKE_WORD:
        return False
    with _wake_pending_lock:
        return time.time() < _wake_pending_until


def _is_wake_active() -> bool:
    if not WAKE_WORD:
        return True
    with _wake_lock:
        return time.time() < _wake_until


def _is_wake_engaged() -> bool:
    if not WAKE_WORD:
        return True
    return _is_wake_pending() or _is_wake_active()


def _set_wake_active() -> None:
    if not WAKE_WORD:
        return
    global _wake_until
    global _wake_pending_until
    with _wake_lock:
        _wake_until = time.time() + WAKE_ACTIVE_SECONDS
    with _wake_pending_lock:
        _wake_pending_until = 0.0


def _set_wake_pending() -> None:
    if not WAKE_WORD:
        return
    if WAKE_PENDING_SECONDS <= 0:
        return
    global _wake_pending_until
    with _wake_pending_lock:
        _wake_pending_until = time.time() + WAKE_PENDING_SECONDS


def _record_chunk(path: str, duration: float) -> None:
    duration_sec = max(1, int(round(duration)))
    command = [
        "arecord",
        "-q",
        "-D",
        AUDIO_DEVICE,
        "-f",
        "S16_LE",
        "-r",
        str(AUDIO_SAMPLE_RATE),
        "-c",
        "1",
        "-d",
        str(duration_sec),
        path,
    ]
    subprocess.run(command, check=True)


def _chunk_rms(path: str) -> int:
    with wave.open(path, "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        if not frames:
            return 0
        return int(audioop.rms(frames, wav.getsampwidth()))


def _combine_chunks(chunks: list[str], out_path: str) -> None:
    if not chunks:
        return

    with wave.open(chunks[0], "rb") as first:
        params = first.getparams()

    with wave.open(out_path, "wb") as out:
        out.setparams(params)
        for chunk in chunks:
            with wave.open(chunk, "rb") as wav:
                out.writeframes(wav.readframes(wav.getnframes()))


def _send_audio(session: requests.Session, wav_path: str, include_wake_word: bool) -> None:
    backoff = max(0.1, STT_RETRY_BACKOFF)
    retries = max(1, STT_RETRIES)
    if include_wake_word:
        _set_wake_pending()

    for attempt in range(1, retries + 1):
        try:
            with open(wav_path, "rb") as audio_file:
                files = {"audio": (os.path.basename(wav_path), audio_file, "audio/wav")}
                data = {}
                if include_wake_word and WAKE_WORD:
                    data["wake_word"] = WAKE_WORD
                if DEVICE_ID:
                    data["device_id"] = DEVICE_ID

                response = session.post(
                    STT_ENDPOINT,
                    files=files,
                    data=data,
                    timeout=HTTP_TIMEOUT,
                )
        except requests.RequestException as exc:
            print(f"[WARN] Audio send failed (attempt {attempt}): {exc}")
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            return

        if response.status_code != 200:
            print(f"[WARN] STT error {response.status_code}: {response.text}")
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            return

        break

    try:
        payload = response.json()
    except ValueError:
        return

    message = payload.get("message") or ""
    transcript = payload.get("transcript") or ""
    if transcript:
        print(f"[INFO] STT transcript: {transcript}")
    wake_triggered = False
    matched_wake = ""
    wake_only = False
    if transcript:
        normalized_transcript = _normalize_text(transcript.lower())
        for wake_word in _wake_words():
            normalized_wake = _normalize_text(wake_word.lower())
            if normalized_wake and normalized_wake in normalized_transcript:
                wake_triggered = True
                matched_wake = wake_word
                break
        wake_only = _is_wake_only(transcript)
    if wake_triggered:
        print(f"[INFO] Wake word detected: {matched_wake or WAKE_WORD}")
        _set_wake_active()
        if wake_only:
            if WAKE_RESPONSE:
                speak(WAKE_RESPONSE)
    if message:
        print(f"[SERVER REPLY] {message}")
        speak(message)


def audio_loop(stop_event: threading.Event) -> None:
    if not shutil.which("arecord"):
        print("[WARN] arecord not available. Audio disabled.")
        return

    session = requests.Session()
    chunks: list[str] = []
    speech_active = False
    silence_elapsed = 0.0

    while not stop_event.is_set():
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk_path = tmp.name

        try:
            _record_chunk(chunk_path, AUDIO_CHUNK_SEC)
        except Exception as exc:
            print(f"[WARN] Audio capture failed: {exc}")
            try:
                os.remove(chunk_path)
            except OSError:
                pass
            time.sleep(1)
            continue

        try:
            rms = _chunk_rms(chunk_path)
        except Exception:
            rms = 0

        has_voice = rms >= AUDIO_RMS_THRESHOLD

        if has_voice:
            speech_active = True
            silence_elapsed = 0.0
            chunks.append(chunk_path)
            continue

        if speech_active:
            silence_elapsed += AUDIO_CHUNK_SEC
            chunks.append(chunk_path)
            if silence_elapsed >= SILENCE_SECONDS:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as merged:
                    merged_path = merged.name
                try:
                    _combine_chunks(chunks, merged_path)
                    _send_audio(session, merged_path, include_wake_word=not _is_wake_engaged())
                finally:
                    for chunk in chunks:
                        try:
                            os.remove(chunk)
                        except OSError:
                            pass
                    try:
                        os.remove(merged_path)
                    except OSError:
                        pass
                chunks = []
                speech_active = False
                silence_elapsed = 0.0
        else:
            try:
                os.remove(chunk_path)
            except OSError:
                pass


async def stream_video(display: DisplayManager) -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Camera not available")
        return

    print(f"[INFO] Camera opened (index={CAMERA_INDEX})")
    frame_interval = 1.0 / FPS if FPS > 0 else 0.0
    ws_url = _resolve_ws_url(DEVICE_ID)

    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                print(f"[INFO] WS connected: {ws_url}")
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

                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=0.001)
                        if message:
                            try:
                                data = json.loads(message)
                            except ValueError:
                                data = {}
                            tts = data.get("tts") or data.get("message")
                            if tts:
                                text = str(tts)
                                print(f"[INFO] TTS message received: {text}")
                                announce_only = bool(data.get("announce_only"))
                                if announce_only:
                                    announcement = (
                                        TTS_NOTIFICATION.strip()
                                        or "\uba54\uc2dc\uc9c0\uac00 \ub3c4\ucc29\ud588\uc5b4\uc694. \ud655\uc778\ud574\uc8fc\uc138\uc694."
                                    )
                                    speak(announcement)
                                else:
                                    speak(text)
                                display.post_message(text, data.get("log_id"))
                    except asyncio.TimeoutError:
                        pass

                    if frame_interval:
                        await asyncio.sleep(frame_interval)
        except Exception as exc:
            print(f"[WARN] WS reconnect: {exc}")
            await asyncio.sleep(2)


def _run_stream(display: DisplayManager) -> None:
    asyncio.run(stream_video(display))


def main() -> None:
    global DEVICE_ID
    DEVICE_ID = _handshake_device()
    if DEVICE_SERIAL:
        print(f"[INFO] Device handshake resolved device_id={DEVICE_ID}")
    display = DisplayManager()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    print("[INFO] Vision client starting")
    if AUDIO_ENABLED:
        audio_thread = threading.Thread(target=audio_loop, args=(stop_event,), daemon=True)
        audio_thread.start()
        threads.append(audio_thread)
    else:
        print("[INFO] Audio disabled.")

    video_thread = threading.Thread(target=_run_stream, args=(display,), daemon=True)
    video_thread.start()
    threads.append(video_thread)

    try:
        display.run()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1)


if __name__ == "__main__":
    main()
