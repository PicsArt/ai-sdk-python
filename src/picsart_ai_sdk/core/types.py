"""Model definition and option types — port of ``src/core/types.ts`` (the runtime parts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

from .descriptors.types import ModelParams

GenerationMode = Literal["video", "image", "audio", "text", "json"]
ReleaseTag = Literal["preview", "production", "general-availability"]
BadgeType = Literal["new", "popular", "coming-soon", "fast", "premium", "hot"]

Constraint = dict[str, Any]
"""A constraint off the wire::

    {"when": {key: {"is": v} | {"exists": bool}},
     "then": {key: {"allowed": [...]} | {"disabled": True} | {"required": True}, "reason"?}}
"""


class ModelFeature(TypedDict):
    label: str
    variant: str


class ParamOption(TypedDict):
    id: str
    label: str


class VoiceOption(TypedDict):
    id: str
    name: str
    description: str
    tags: list[str]
    provider: str
    previewUrl: NotRequired[str]


class AvatarOption(TypedDict):
    id: str
    name: str
    description: str
    tags: list[str]
    provider: str
    previewImageUrl: NotRequired[str]
    previewVideoUrl: NotRequired[str]
    gender: NotRequired[str]
    defaultVoiceId: NotRequired[str]


@dataclass
class ModelDefinition:
    """A catalog model (TS ``ModelDefinition``; builders and output schemas do not travel on the wire)."""

    id: str
    name: str
    provider: str
    provider_name: str
    provider_color: str
    provider_label: str
    description: str
    mode: str
    input_type: str
    workflow: str
    param_config: ModelParams = field(default_factory=dict)
    features: list[ModelFeature] = field(default_factory=list)
    badge: list[str] | None = None
    release: str | None = None
    added_at: str | None = None
    estimated_time: float | dict[str, float] | None = None
    deprecated: bool | None = None
    constraints: list[Constraint] | None = None
    edit_workflow: str | None = None
    model_id: str | None = None
    sync_execute: bool | None = None
    output_extension: str | None = None
