"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.
"""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

django_asgi_app = get_asgi_application()

import dashboard.routing

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(
            URLRouter(dashboard.routing.websocket_urlpatterns)
        ),
    }
)
