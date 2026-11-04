"""Model id + context → payload — port of ``src/core/deeplink/serialize.ts``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..descriptors.utils import extract_defaults
from ..js import strict_equal, truthy
from ..model_registry import get_model
from .codec import encode

DEDICATED_KEYS = frozenset({"prompt", "negativePrompt", "imageUrls", "videoUrl", "audioUrl", "startFrame", "endFrame"})
"""Context keys carried as dedicated payload fields (not in ``o``)."""

EXCLUDED_KEYS = frozenset(
    {
        "audioFile",
        "mentionedPersonas",
        "modelId",
        "exploreImageId",
        "sourceImageId",
        "sourceImageUrl",
        "multiPrompt",
        "elementList",
        "voiceList",
        "dynamicMasks",
        "staticMask",
        "callbackUrl",
        "externalTaskId",
    }
)
"""Keys never written to ``o`` (not serializable or out of scope)."""

_FIELDS = (
    ("prompt", "p"),
    ("negativePrompt", "np"),
    ("imageUrls", "i"),
    ("videoUrl", "vi"),
    ("audioUrl", "au"),
    ("startFrame", "sf"),
    ("endFrame", "ef"),
)


def _has_length(value: Any) -> bool:
    """``value?.length`` is truthy."""
    return isinstance(value, (list, tuple, str)) and len(value) > 0


def serialize_payload(model_id: str, context: Mapping[str, Any]) -> str:
    """Encode a model id + context; ``o`` holds only params that differ from the model's defaults."""
    payload: dict[str, Any] = {"v": 1, "m": model_id}
    for key, field in _FIELDS:
        value = context.get(key)
        if _has_length(value) if key == "imageUrls" else truthy(value):
            payload[field] = value

    model = get_model(model_id)
    defaults = extract_defaults(model.param_config) if model is not None else {}
    overrides: dict[str, Any] = {}
    for key, value in context.items():
        if key in DEDICATED_KEYS or key in EXCLUDED_KEYS or value is None:
            continue
        if key in defaults and strict_equal(defaults[key], value):
            continue
        overrides[key] = value
    if overrides:
        payload["o"] = overrides
    return encode(payload)
