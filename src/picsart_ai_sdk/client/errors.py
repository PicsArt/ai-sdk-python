"""Workflows-client errors → :class:`ApiError` — port of ``src/client/workflows-error.ts``."""

from __future__ import annotations

import re

from ..abort import AbortError
from ..core.errors import ApiError, code_for_status

_GENERIC_REASONS = {"request_failed", "unknown_error"}


def _is_workflows_error(err: BaseException) -> bool:
    name = getattr(err, "name", type(err).__name__)
    return isinstance(getattr(err, "reason", None), str) and bool(re.search(r"WorkflowsError$", str(name)))


def to_api_error(err: BaseException, workflow: str, task_id: str | None = None) -> BaseException:
    """Map anything raised while talking to the platform to an :class:`ApiError` (aborts pass through)."""
    if isinstance(err, (ApiError, AbortError)):
        return err
    if _is_workflows_error(err):
        reason: str = getattr(err, "reason", "")
        if reason == "client_timeout":
            return ApiError(
                f"Timed out waiting for workflow {workflow}{f':{task_id}' if task_id else ''}",
                status=408,
                code="timeout",
            )
        status_code = getattr(err, "http_status_code", None)
        status = 502 if status_code is None else int(status_code)
        message = getattr(err, "message", None)
        return ApiError(
            "Request failed" if message is None else str(message),
            status=status,
            code=reason if reason and reason not in _GENERIC_REASONS else code_for_status(status),
        )
    return ApiError(str(err), status=502, code="generation_failed")
