"""ASGI middleware for request-size limits and safe access logging.

Responsibility
    Reject oversized request bodies before FastAPI parses them. A declared ``Content-Length``
    is checked first; the middleware also counts the bytes actually received so a missing,
    incorrect or chunked length cannot bypass the limit. The body is buffered only up to the
    configured maximum and is replayed to the application unchanged.

    The access logger records only the method, path, status and duration. It never logs query
    strings, headers, bearer tokens or request bodies.

Limits
    This is a JSON API with bounded payloads, not a streaming upload service. Buffering one
    request body up to ``max_body_bytes`` is deliberate; anything larger gets a stable 413
    response and never reaches a route handler.
"""

from __future__ import annotations

import logging
import time

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class AccessLogMiddleware:
    """Log method, path, status and duration only.

    Query strings, headers, tokens and request bodies are deliberately never logged. The path
    is rendered with ``repr`` so a hostile newline cannot forge an additional log line.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "%s %r -> %d (%.1f ms)",
                scope.get("method", "?"),
                scope.get("path", ""),
                status_code,
                duration_ms,
            )


class RequestSizeLimitMiddleware:
    """Reject HTTP request bodies larger than ``max_body_bytes``."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if max_body_bytes < 1:
            msg = "max_body_bytes must be positive"
            raise ValueError(msg)
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                declared_length = int(declared)
            except ValueError:
                await self._reject(scope, receive, send, 400, "invalid Content-Length header")
                return
            if declared_length < 0:
                await self._reject(scope, receive, send, 400, "invalid Content-Length header")
                return
            if declared_length > self.max_body_bytes:
                await self._reject(scope, receive, send, 413, "request body too large")
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await self._reject(scope, receive, send, 413, "request body too large")
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        detail: str,
    ) -> None:
        response = JSONResponse(status_code=status_code, content={"detail": detail})
        await response(scope, receive, send)


__all__ = ["AccessLogMiddleware", "RequestSizeLimitMiddleware"]
