"""ASGI middleware tests for request-size enforcement and safe access logging.

The middleware is driven directly with a minimal ASGI harness. No HTTP server, database or
model is involved, which keeps the boundary behaviour deterministic and fast.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Coroutine
from typing import Any

import pytest
from starlette.types import Message, Receive, Scope, Send

from fraud_risk.api.middleware import AccessLogMiddleware, RequestSizeLimitMiddleware


def http_scope(
    *,
    method: str = "POST",
    path: str = "/predict",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Scope:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"token=must-not-be-logged",
        "headers": headers or [],
    }


def receive_from(messages: list[Message]) -> Receive:
    async def receive() -> Message:
        return messages.pop(0)

    return receive


async def send_into(messages: list[Message]) -> Send:
    async def send(message: Message) -> None:
        messages.append(message)

    return send


def run(coro: Coroutine[Any, Any, None]) -> None:
    asyncio.run(coro)


def response_status(messages: list[Message]) -> int:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return int(start["status"])


def response_body(messages: list[Message]) -> dict[str, Any]:
    body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    decoded = json.loads(body.decode("utf-8"))
    assert isinstance(decoded, dict)
    return decoded


def test_request_size_limit_requires_a_positive_limit() -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive, send

    with pytest.raises(ValueError, match="positive"):
        RequestSizeLimitMiddleware(app, max_body_bytes=0)


def test_non_http_scopes_pass_through_both_middlewares() -> None:
    called: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del receive, send
        called.append(str(scope["type"]))

    scope: Scope = {"type": "lifespan"}

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:
        del message

    run(RequestSizeLimitMiddleware(app, max_body_bytes=1)(scope, receive, send))
    run(AccessLogMiddleware(app)(scope, receive, send))

    assert called == ["lifespan", "lifespan"]


@pytest.mark.parametrize(
    ("header", "status", "detail"),
    [
        (b"abc", 400, "invalid Content-Length header"),
        (b"-1", 400, "invalid Content-Length header"),
        (b"11", 413, "request body too large"),
    ],
)
def test_declared_content_length_is_validated_before_reading(
    header: bytes,
    status: int,
    detail: str,
) -> None:
    called = False

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal called
        del scope, receive, send
        called = True

    messages: list[Message] = []

    async def scenario() -> None:
        middleware = RequestSizeLimitMiddleware(app, max_body_bytes=10)
        send = await send_into(messages)
        await middleware(
            http_scope(headers=[(b"content-length", header)]),
            receive_from([{"type": "http.request", "body": b"ignored"}]),
            send,
        )

    run(scenario())

    assert called is False
    assert response_status(messages) == status
    assert response_body(messages) == {"detail": detail}


def test_chunked_body_is_counted_and_rejected_when_it_exceeds_the_limit() -> None:
    called = False

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal called
        del scope, receive, send
        called = True

    messages: list[Message] = []

    async def scenario() -> None:
        middleware = RequestSizeLimitMiddleware(app, max_body_bytes=5)
        await middleware(
            http_scope(),
            receive_from(
                [
                    {"type": "http.request", "body": b"1234", "more_body": True},
                    {"type": "http.request", "body": b"5678", "more_body": False},
                ]
            ),
            await send_into(messages),
        )

    run(scenario())

    assert called is False
    assert response_status(messages) == 413
    assert response_body(messages) == {"detail": "request body too large"}


def test_body_within_the_limit_is_buffered_and_replayed_once() -> None:
    received: list[bytes] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope
        first = await receive()
        second = await receive()
        received.append(first.get("body", b""))
        assert second["type"] == "http.disconnect"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    messages: list[Message] = []

    async def scenario() -> None:
        middleware = RequestSizeLimitMiddleware(app, max_body_bytes=10)
        await middleware(
            http_scope(),
            receive_from(
                [
                    {"type": "http.request", "body": b"hello", "more_body": True},
                    {"type": "http.request", "body": b"!", "more_body": False},
                ]
            ),
            await send_into(messages),
        )

    run(scenario())

    assert received == [b"hello!"]
    assert response_status(messages) == 200


def test_disconnect_while_reading_returns_without_calling_the_app_or_sending() -> None:
    called = False

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal called
        del scope, receive, send
        called = True

    messages: list[Message] = []

    async def scenario() -> None:
        middleware = RequestSizeLimitMiddleware(app, max_body_bytes=10)
        await middleware(
            http_scope(),
            receive_from([{"type": "http.disconnect"}]),
            await send_into(messages),
        )

    run(scenario())

    assert called is False
    assert messages == []


def test_access_log_contains_only_method_path_status_and_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive
        await send({"type": "http.response.start", "status": 418, "headers": []})
        await send({"type": "http.response.body", "body": b"private-body"})

    messages: list[Message] = []
    scope = http_scope(
        method="POST",
        path="/predict",
        headers=[(b"authorization", b"Bearer secret-token")],
    )

    async def scenario() -> None:
        await AccessLogMiddleware(app)(scope, receive_from([]), await send_into(messages))

    with caplog.at_level(logging.INFO, logger="fraud_risk.api.middleware"):
        run(scenario())

    assert response_status(messages) == 418
    assert "POST '/predict' -> 418" in caplog.text
    assert "must-not-be-logged" not in caplog.text
    assert "secret-token" not in caplog.text
    assert "private-body" not in caplog.text


def test_access_log_still_records_a_500_when_the_app_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive, send
        raise RuntimeError("synthetic failure")

    async def receive() -> Message:
        return {"type": "http.request", "body": b""}

    async def send(message: Message) -> None:
        del message

    with (
        caplog.at_level(logging.INFO, logger="fraud_risk.api.middleware"),
        pytest.raises(RuntimeError, match="synthetic failure"),
    ):
        run(AccessLogMiddleware(app)(http_scope(path="/explode"), receive, send))

    assert "POST '/explode' -> 500" in caplog.text
    assert "synthetic failure" not in caplog.text
