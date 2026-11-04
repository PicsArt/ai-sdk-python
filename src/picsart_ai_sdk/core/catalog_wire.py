"""The models-catalog wire format — port of ``src/core/catalog-wire.ts``."""

from __future__ import annotations

from typing import Any

from .descriptors.types import ParamEntry
from .errors import ApiError
from .js import truthy
from .types import ModelDefinition

ENTRY_META_KEYS = ("label", "required", "category")


def _to_param_entry(wire: dict[str, Any]) -> ParamEntry:
    descriptor = {k: v for k, v in wire.items() if k not in ENTRY_META_KEYS}
    return ParamEntry(
        descriptor=descriptor, label=wire.get("label"), required=wire.get("required"), category=wire.get("category")
    )


def _to_catalog_param(entry: ParamEntry) -> dict[str, Any]:
    wire = dict(entry.descriptor)
    for key in ENTRY_META_KEYS:
        value = getattr(entry, key)
        if value is not None:
            wire[key] = value
    return wire


def _invalid_response(detail: str) -> ApiError:
    return ApiError(f"Invalid models-catalog response: {detail}", status=502, code="invalid_response")


def _to_model_definition(wire: Any) -> ModelDefinition:
    # JS `!wire.provider`: `{}` / `[]` are truthy; a non-object provider reads as all-undefined fields.
    if not isinstance(wire, dict) or not isinstance(wire.get("id"), str) or not truthy(wire.get("provider")):
        raise _invalid_response("model entry is missing `id` or `provider`")
    provider: Any = wire["provider"] if isinstance(wire["provider"], dict) else {}
    return ModelDefinition(
        id=wire["id"],
        name=wire.get("name"),  # type: ignore[arg-type]
        provider=provider.get("id"),
        provider_name=provider.get("name"),
        provider_color=provider.get("color"),
        provider_label=provider.get("label"),
        description=wire["description"] if wire.get("description") is not None else "",
        features=wire["features"] if wire.get("features") is not None else [],
        mode=wire.get("mode"),  # type: ignore[arg-type]
        input_type=wire.get("inputType"),  # type: ignore[arg-type]
        workflow=wire["workflow"] if wire.get("workflow") is not None else "",
        param_config={key: _to_param_entry(param) for key, param in (wire.get("params") or {}).items()},
        badge=wire.get("badges"),
        release=wire.get("release"),
        added_at=wire.get("addedAt"),
        estimated_time=wire.get("estimatedTime"),
        deprecated=wire.get("deprecated"),
        constraints=wire.get("constraints"),
        edit_workflow=wire.get("editWorkflow"),
        model_id=wire.get("modelId"),
        sync_execute=wire.get("syncExecute"),
    )


def deserialize_catalog(body: Any) -> list[ModelDefinition]:
    """The models of a ``/v1/models-catalog`` response; 502 ``invalid_response`` on any other shape."""
    response = body.get("response") if isinstance(body, dict) else None
    if (
        not isinstance(body, dict)
        or body.get("status") != "success"
        or not isinstance(response, dict)
        or not isinstance(response.get("models"), list)
    ):
        raise _invalid_response('expected `{ status: "success", response: { models: [...] } }`')
    return [_to_model_definition(m) for m in response["models"]]


def serialize_catalog(models: list[ModelDefinition]) -> dict[str, Any]:
    """The exact inverse of :func:`deserialize_catalog` (fixtures, round trips)."""
    wire_models = []
    for d in models:
        wire: dict[str, Any] = {
            "id": d.id,
            "name": d.name,
            "description": d.description,
            "mode": d.mode,
            "inputType": d.input_type,
            "provider": {
                "id": d.provider,
                "name": d.provider_name,
                "color": d.provider_color,
                "label": d.provider_label,
            },
            "features": d.features,
            "params": {key: _to_catalog_param(entry) for key, entry in d.param_config.items()},
        }
        for key, value in (
            ("badges", d.badge),
            ("release", d.release),
            ("addedAt", d.added_at),
            ("estimatedTime", d.estimated_time),
            ("deprecated", d.deprecated),
            ("constraints", d.constraints),
        ):
            if value is not None:
                wire[key] = value
        if d.workflow:
            wire["workflow"] = d.workflow
        for key, value in (("editWorkflow", d.edit_workflow), ("modelId", d.model_id), ("syncExecute", d.sync_execute)):
            if value is not None:
                wire[key] = value
        wire_models.append(wire)
    return {"status": "success", "response": {"total": len(wire_models), "models": wire_models}}
