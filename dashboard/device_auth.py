import hashlib
import secrets

from django.utils import timezone
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.permissions import BasePermission

from .models import DeviceToken


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class DeviceTokenAuthentication(BaseAuthentication):
    def authenticate(self, request):
        token = self._extract_token(request)
        if not token:
            return None

        token_hash = hash_token(token)
        try:
            token_obj = DeviceToken.objects.select_related("device", "device__owner").get(
                token_hash=token_hash,
                revoked=False,
                device__is_active=True,
            )
        except DeviceToken.DoesNotExist:
            raise AuthenticationFailed("Invalid device token.")

        request.device = token_obj.device
        token_obj.device.last_seen_at = timezone.now()
        token_obj.device.save(update_fields=["last_seen_at"])
        return (None, token_obj)

    @staticmethod
    def _extract_token(request) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("device "):
            return auth_header.split(" ", 1)[1].strip()

        return request.headers.get("X-Device-Token", "").strip()


class DeviceAuthRequired(BasePermission):
    def has_permission(self, request, view):
        return hasattr(request, "device")
