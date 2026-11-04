"""Result helpers the v7 runtime uses — port of ``src/core/response.ts`` (``throwIfErrorResult``,
``toCompletedStatus``, ``isSeedanceDraftTask``)."""

from __future__ import annotations

from typing import Any, TypedDict

from .errors import ApiError, code_for_status
from .js import is_number, js_string, nullish, truthy
from .workflow import CreditUsage, WorkflowJobHandle, WorkflowStatusResult


class SeedanceDraftTask(TypedDict):
    id: str


class GenerateResultItemMetadata(TypedDict, total=False):
    exploreImageId: str
    generatedVoiceId: str
    lastFrameUrl: str
    draftTask: SeedanceDraftTask


def throw_if_error_result(result: object, model_name: str) -> None:
    """Raise :class:`ApiError` when a result object is an error payload (a 200 carrying a failure)."""
    if not isinstance(result, dict):
        return
    status = nullish(result.get("status"), result.get("statusCode"))
    message = nullish(result.get("message"), result.get("error"), result.get("reason"))
    is_error = (is_number(status) and status >= 400) or status in ("error", "FAILED")
    if not is_error:
        return
    numeric = is_number(status)
    code_text = f" ({js_string(status)})" if numeric else ""
    detail = js_string(message) if truthy(message) else "unknown error"
    http_status = int(status) if numeric else 502
    reason = result.get("reason")
    raise ApiError(
        f"{model_name} failed{code_text}: {detail}",
        status=http_status,
        code=reason if isinstance(reason, str) and reason else code_for_status(http_status),
    )


def to_completed_status(
    handle: WorkflowJobHandle, result: Any, raw: Any, usage: CreditUsage | None = None
) -> WorkflowStatusResult:
    return WorkflowStatusResult(handle=handle, status="COMPLETED", result=result, raw=raw, usage=usage)


def is_seedance_draft_task(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("id"), str)
