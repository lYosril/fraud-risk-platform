"""Safe formatting for request and per-item validation failures.

Responsibility
    Turn a Pydantic/FastAPI validation exception into the short error string the API exposes.
    The raw validation payload contains the submitted value under ``input``; that must never
    reach the response because it can include a feature snapshot or other request data.
"""

from __future__ import annotations

from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError


def safe_validation_detail(exc: ValidationError | RequestValidationError) -> str:
    """Return the first error's location and message without echoing its input value."""
    errors = exc.errors()
    if not errors:
        return "invalid request"

    first = errors[0]
    location = ".".join(str(part) for part in first["loc"] if part not in {"body", "query", "path"})
    message = str(first["msg"])
    return f"{location}: {message}" if location else message


__all__ = ["safe_validation_detail"]
