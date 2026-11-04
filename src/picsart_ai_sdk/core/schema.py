"""The flat param schema shape — port of ``src/core/schema.ts`` (what ``to_schema()`` returns per key)."""

from __future__ import annotations

from typing import Literal, TypedDict


class ParamSchema(TypedDict, total=False):
    type: Literal["string", "number", "boolean", "file"]
    enum: list[str | int | float]
    default: str | int | float | bool
    min: float
    max: float
    step: float
    required: bool
    label: str
    accept: str


ModelParamSchema = dict[str, ParamSchema]
