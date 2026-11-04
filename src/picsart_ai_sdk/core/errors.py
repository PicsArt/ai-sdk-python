"""Unified SDK error — port of ``src/core/errors.ts``.

Every failure the SDK raises from the generation surface, the async lifecycle,
``ai.catalogs`` and ``ai.apis`` is an :class:`ApiError` carrying the same
fields as the TypeScript SDK, so callers branch on ``status`` / ``code``
instead of matching ``message``.
"""

from __future__ import annotations

ApiErrorCode = str
"""Known values: ``unknown_model``, ``wrong_model_mode``, ``validation_error``,
``unsupported_transport``, ``timeout``, ``aborted``, ``canceled``,
``generation_failed``, ``invalid_response``, the status slugs of
:func:`code_for_status`, ``catalog_not_loaded``, ``catalog_conflict``, or any
platform ``reason``."""

_CODE_BY_STATUS: dict[int, ApiErrorCode] = {
    400: "bad_request",
    401: "unauthorized",
    402: "payment_required",
    403: "forbidden",
    404: "not_found",
    408: "timeout",
    409: "conflict",
    413: "payload_too_large",
    422: "unprocessable_entity",
    429: "rate_limited",
    500: "server_error",
    502: "bad_gateway",
    503: "service_unavailable",
    504: "gateway_timeout",
}


class ApiError(Exception):
    """An SDK failure: ``message``, HTTP-like ``status``, machine ``code`` (``reason`` is the same)."""

    name = "ApiError"

    def __init__(self, message: str, *, status: int, code: ApiErrorCode) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.reason = code

    def __reduce__(self) -> tuple[object, ...]:
        return (_rebuild_api_error, (self.message, self.status, self.code))

    def __repr__(self) -> str:
        return f"ApiError({self.message!r}, status={self.status}, code={self.code!r})"


def _rebuild_api_error(message: str, status: int, code: ApiErrorCode) -> ApiError:
    return ApiError(message, status=status, code=code)


def code_for_status(status: int) -> ApiErrorCode:
    """The conventional code for an HTTP status (used when the platform sent no ``reason``)."""
    if status in _CODE_BY_STATUS:
        return _CODE_BY_STATUS[status]
    return "server_error" if status >= 500 else f"http_{status}"
