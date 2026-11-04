"""Descriptor validation, defaults, schema and value transfer — port of ``src/core/descriptors/utils.ts``."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..constraints import evaluate_requirements
from ..js import is_number, js_includes, js_string, js_trim, truthy, utf16_len, utf16_slice
from ..types import Constraint
from .types import ModelParams, ParamDescriptor


class DescriptorError(Exception):
    """A validation failure (a plain ``Error`` in TS); ``ModelDescriptor.validate`` reports its message."""


def extract_defaults(params: ModelParams) -> dict[str, Any]:
    return {key: entry.descriptor["default"] for key, entry in params.items() if "default" in entry.descriptor}


def validate_descriptor(key: str, d: ParamDescriptor, val: Any, required: bool | None = None) -> None:
    if (
        val is None
        or (required and isinstance(val, str) and len(js_trim(val)) == 0)
        or (required and isinstance(val, list) and len(val) == 0)
    ):
        if required:
            raise DescriptorError(f'"{key}" is required')
        if val is None:
            return
    kind = d.get("kind")
    if kind == "enum":
        ids = [o.get("id") for o in d.get("options", [])]
        if not js_includes(ids, val):
            raise DescriptorError(f'"{key}" must be one of: {", ".join(js_string(i) for i in ids)}')
    elif kind == "catalog":
        if not isinstance(val, str):
            raise DescriptorError(f'"{key}" must be a string')
    elif kind == "range":
        if not is_number(val) or (isinstance(val, float) and math.isnan(val)):
            raise DescriptorError(f'"{key}" must be a number')
        if val < d["min"] or val > d["max"]:
            raise DescriptorError(f'"{key}" must be between {js_string(d["min"])} and {js_string(d["max"])}')
    elif kind == "boolean":
        if not isinstance(val, bool):
            raise DescriptorError(f'"{key}" must be a boolean')
    elif kind == "text":
        if not isinstance(val, str):
            raise DescriptorError(f'"{key}" must be a string')
        if d.get("minLength") and utf16_len(js_trim(val)) < d["minLength"]:
            raise DescriptorError(f'"{key}" must be at least {js_string(d["minLength"])} characters')
        if d.get("maxLength") and utf16_len(val) > d["maxLength"]:
            raise DescriptorError(f'"{key}" exceeds max length of {js_string(d["maxLength"])}')
    elif kind == "file":
        array: Any = d.get("array")
        if truthy(array):
            if not isinstance(val, list):
                raise DescriptorError(f'"{key}" must be an array of URLs')
            if array.get("min") is not None and len(val) < array["min"]:
                raise DescriptorError(f'"{key}" needs at least {js_string(array["min"])} items')
            if array.get("max") is not None and len(val) > array["max"]:
                raise DescriptorError(f'"{key}" allows at most {js_string(array["max"])} items')
        elif not isinstance(val, str):
            raise DescriptorError(f'"{key}" must be a string URL')
    elif kind == "object":
        _validate_object(key, d, val, required)


def _validate_object(key: str, d: ParamDescriptor, val: Any, required: bool | None) -> None:
    array: Any = d.get("array")
    has_array = truthy(array)
    if has_array and not isinstance(val, list):
        raise DescriptorError(f'"{key}" must be an array')
    items = val if has_array else [val]
    if has_array and array.get("min") is not None and len(items) < array["min"]:
        raise DescriptorError(f'"{key}" needs at least {js_string(array["min"])} items')
    if has_array and array.get("max") is not None and len(items) > array["max"]:
        raise DescriptorError(f'"{key}" allows at most {js_string(array["max"])} items')
    declared: dict[str, Any] = d.get("fields") or {}
    extra: Any = d.get("additionalProperties")
    for item in items:
        if not isinstance(item, dict):
            raise DescriptorError(f'"{key}" items must be objects' if has_array else f'"{key}" must be an object')
        if required and not has_array and truthy(extra) and not declared and not item:
            raise DescriptorError(f'"{key}" is required')
        for field_key, field_d in declared.items():
            validate_descriptor(f"{key}.{field_key}", field_d, item.get(field_key), field_d.get("required"))
        if truthy(extra):
            for extra_key, extra_val in item.items():
                if extra_key in declared:
                    continue
                validate_descriptor(f"{key}.{extra_key}", extra, extra_val, True)


def _is_absent(val: Any) -> bool:
    return val is None or (isinstance(val, str) and len(js_trim(val)) == 0) or (isinstance(val, list) and len(val) == 0)


def validate_all(
    params: ModelParams,
    values: dict[str, Any],
    constraints: list[Constraint] | None = None,
    validate_input: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Raise :class:`DescriptorError` for the first invalid param, then the first unmet conditional requirement."""
    for key, entry in params.items():
        validate_descriptor(key, entry.descriptor, values.get(key), entry.required)
    for key, requirement in evaluate_requirements(constraints, values).items():
        if _is_absent(values.get(key)):
            raise DescriptorError(
                f'"{key}" is required: {requirement.reason}' if requirement.reason else f'"{key}" is required'
            )
    if validate_input is not None:
        validate_input(values)


def descriptors_to_schema(params: ModelParams) -> dict[str, dict[str, Any]]:
    """The flat ``ModelParamSchema`` (enum / range / boolean / text / catalog / file; others omitted).

    A field TS leaves ``undefined`` (absent on the descriptor, unset on the entry) is left out, so the schema
    serializes exactly as ``JSON.stringify`` writes the TS one.
    """

    def pick(source: dict[str, Any], *keys: str) -> dict[str, Any]:
        return {key: source[key] for key in keys if key in source}

    def meta(**fields: Any) -> dict[str, Any]:
        return {key: value for key, value in fields.items() if value is not None}

    schema: dict[str, dict[str, Any]] = {}
    for key, entry in params.items():
        d = entry.descriptor
        kind = d.get("kind")
        if kind == "enum":
            schema[key] = {
                **({"type": d["valueType"]} if "valueType" in d else {}),
                "enum": [o.get("id") for o in d.get("options", [])],
                **pick(d, "default"),
            }
        elif kind == "range":
            schema[key] = {"type": "number", **pick(d, "min", "max", "step", "default")}
        elif kind == "boolean":
            schema[key] = {"type": "boolean", **pick(d, "default")}
        elif kind == "text":
            schema[key] = {"type": "string", **meta(label=entry.label)}
        elif kind == "catalog":
            schema[key] = {"type": "string", **pick(d, "default"), **meta(label=entry.label)}
        elif kind == "file":
            schema[key] = {"type": "file", **meta(required=entry.required, label=entry.label), **pick(d, "accept")}
    return schema


_UNSET = object()


def _coerce_array_shape(prev: Any, array: dict[str, Any] | None, is_empty: Callable[[Any], bool]) -> Any:
    if truthy(array):
        assert array is not None
        maximum = array.get("max")
        items = prev if isinstance(prev, list) else [prev]
        trimmed = [v for v in items if not is_empty(v)]
        if maximum is not None:
            trimmed = trimmed[: max(0, int(maximum))]
        return trimmed if trimmed else _UNSET
    single = (prev[0] if prev else None) if isinstance(prev, list) else prev
    return _UNSET if is_empty(single) else single


def transfer_values(new_params: ModelParams, prev: dict[str, Any]) -> dict[str, Any]:
    """Carry values to another model: start from its defaults, keep what fits, drop unknown keys."""
    ctx = extract_defaults(new_params)
    for key, entry in new_params.items():
        prev_val = prev.get(key)
        if prev_val is None:
            continue
        d = entry.descriptor
        kind = d.get("kind")
        if kind == "enum":
            if js_includes([o.get("id") for o in d.get("options", [])], prev_val):
                ctx[key] = prev_val
        elif kind == "catalog":
            pass
        elif kind == "range":
            if is_number(prev_val):
                ctx[key] = min(max(prev_val, d["min"]), d["max"])
        elif kind == "boolean":
            if isinstance(prev_val, bool):
                ctx[key] = prev_val
        elif kind == "text":
            if isinstance(prev_val, str):
                ctx[key] = utf16_slice(prev_val, d["maxLength"]) if d.get("maxLength") is not None else prev_val
        elif kind == "file":
            nxt = _coerce_array_shape(prev_val, d.get("array"), lambda v: v is None or v == "")
            if nxt is not _UNSET:
                ctx[key] = nxt
        elif kind == "object":
            nxt = _coerce_array_shape(prev_val, d.get("array"), lambda v: v is None)
            if nxt is not _UNSET:
                ctx[key] = nxt
        else:
            ctx[key] = prev_val
    return ctx
