"""Descriptor shapes — port of ``src/core/descriptors/types.ts``.

Descriptors come off the catalog wire, so they are dicts with the TS keys
(``kind``, ``valueType``, ``maxLength``…). What the SDK builds — the model
definition's ``ParamEntry``, ``ModelMeta``, ``ValidationResult``, credit
ranges — are frozen dataclasses with snake_case fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

ParamDescriptor = dict[str, Any]
"""One of the kinds ``enum``, ``catalog``, ``range``, ``boolean``, ``text``, ``file``, ``object``, ``unknown``."""

FlatParamEntry = dict[str, Any]
"""``{...meta, ...descriptor}`` (plus ``key`` from ``all()``) — what the params accessor returns."""

Category = Literal["asset", "reference"]


class EnumOption(TypedDict):
    id: str | float
    label: NotRequired[str]
    disabled: NotRequired[bool]
    disabledReason: NotRequired[str]


@dataclass
class ParamEntry:
    """A model's param: entry meta plus its descriptor (``catalog_options`` holds a hydrated catalog)."""

    descriptor: ParamDescriptor
    label: str | None = None
    required: bool | None = None
    category: Category | None = None
    catalog_options: list[Any] | None = None


ModelParams = dict[str, ParamEntry]


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str
    color: str
    label: str


@dataclass(frozen=True)
class ModelMeta:
    mode: str
    input_type: str
    description: str
    features: list[dict[str, Any]]
    badges: list[str]
    provider: ProviderInfo
    added_at: str | None
    release: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str] | None = None


@dataclass(frozen=True)
class CreditTier:
    credits: float
    unit: str
    quality: str | None = None
    audio: bool | None = None
    use_case: str | None = None


@dataclass(frozen=True)
class CreditRange:
    min: float
    max: float
    tiers: list[CreditTier] = field(default_factory=list)
    unit: str | None = None


@dataclass(frozen=True)
class CreditRangeContext:
    resolution: str | None = None
    generate_audio: bool | None = None


@dataclass(frozen=True)
class ModelFilter:
    output: str | None = None
    provider: str | None = None
    release: list[str] | None = None
