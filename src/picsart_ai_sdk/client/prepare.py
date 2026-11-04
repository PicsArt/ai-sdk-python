"""The ``v1/models`` command and its result parsers — port of ``src/client/prepare.ts``.

Every generation rides one workflow: the SDK sends ``{model, params, options}``
and models-hub builds the vendor payload and normalizes the result to
``{url?, items?, text?, model?}``.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ApiError
from ..core.js import nullish, truthy
from ..core.response import throw_if_error_result
from ..core.workflow import WorkflowStatusResult
from .types import GenerateResult, GenerateResultItem, GenerateTextResult, RunResponse

MODELS_WORKFLOW = "v1/models"


def build_models_command(
    model: str,
    params: dict[str, Any],
    *,
    with_options: bool = False,
    drive: dict[str, Any] | None = None,
    downscale_oversized_images: bool | None = None,
) -> dict[str, Any]:
    """``{model, params}``; media generations add ``options`` (inputs transformation, optional drive)."""
    if not with_options:
        return {"model": model, "params": params}
    options: dict[str, Any] = {
        "inputs_transformation": {"downscale_oversized_images": bool(downscale_oversized_images)},
    }
    if drive is not None:  # JS: any object, even {}, is truthy
        options["drive"] = drive
    return {"model": model, "params": params, "options": options}


def _as_models_result(result: object) -> dict[str, Any]:
    return result if isinstance(result, dict) else {}


def parse_dynamic_result(completed: WorkflowStatusResult, model_ref: str) -> GenerateResult:
    throw_if_error_result(completed.result, model_ref)
    r = _as_models_result(completed.result)
    raw_items = r.get("items")
    items: list[GenerateResultItem]
    if isinstance(raw_items, list) and raw_items:
        items = [
            GenerateResultItem(
                url=item.get("url") if isinstance(item, dict) else None,  # type: ignore[arg-type]
                metadata=item.get("metadata") if isinstance(item, dict) and truthy(item.get("metadata")) else None,
            )
            for item in raw_items
        ]
    elif truthy(r.get("url")):
        items = [GenerateResultItem(url=r["url"])]
    else:
        items = []
    if not items or not truthy(items[0].url):
        raise ApiError(f"{model_ref}: unexpected response — no result URL", status=502, code="invalid_response")
    return GenerateResult(
        url=items[0].url,
        items=items,
        results=items,
        generation_id=completed.handle.id or None,
        usage=completed.usage,
    )


def parse_dynamic_text_result(completed: WorkflowStatusResult, model_ref: str) -> GenerateTextResult:
    throw_if_error_result(completed.result, model_ref)
    throw_if_error_result(completed.raw, model_ref)
    r = _as_models_result(completed.result)
    text = r.get("text")
    if not isinstance(text, str):
        raise ApiError(f"{model_ref}: unexpected response — no text", status=502, code="invalid_response")
    return GenerateTextResult(
        text=text,
        model=nullish(r.get("model"), model_ref),
        raw=nullish(completed.raw, completed.result),
        usage=completed.usage,
    )


def parse_dynamic_run_result(completed: WorkflowStatusResult, model_ref: str) -> RunResponse[Any]:
    throw_if_error_result(completed.result, model_ref)
    throw_if_error_result(completed.raw, model_ref)
    if completed.result is None:
        raise ApiError(f"{model_ref}: unexpected response — empty result", status=502, code="invalid_response")
    return RunResponse(result=completed.result, usage=completed.usage, id=completed.handle.id or None)
