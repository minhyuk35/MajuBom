import os
import random
import time

import requests

API_URL = os.getenv("ENV_API_URL", "http://192.168.0.30:8000/api/environment/")
INTERVAL = float(os.getenv("ENV_INTERVAL", "10"))
TIMEOUT = float(os.getenv("ENV_TIMEOUT", "10"))
SOURCE = os.getenv("ENV_SOURCE", "simulator")


def generate_value(min_value: float, max_value: float) -> float:
    return round(random.uniform(min_value, max_value), 1)


def main() -> None:
    while True:
        payload = {
            "temperature": generate_value(20.0, 30.0),
            "humidity": generate_value(40.0, 60.0),
            "source": SOURCE,
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            print(f"[OK] {payload}")
        except requests.RequestException as exc:
            print(f"[WARN] {exc}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
