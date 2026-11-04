"""Payload → model id + context — port of ``src/core/deeplink/deserialize.ts``."""

from __future__ import annotations

from typing import Any

from ..descriptors.utils import transfer_values
from ..js import js_string, strict_equal, truthy
from ..model_registry import get_model
from .codec import decode
from .sanitize import is_allowed_url, sanitize_prompt
from .types import DeepLinkResult

_MEDIA = (
    ("vi", "videoUrl", "Rejected video URL"),
    ("au", "audioUrl", "Rejected audio URL"),
    ("sf", "startFrame", "Rejected start frame URL"),
    ("ef", "endFrame", "Rejected end frame URL"),
)
# Context keys that only their dedicated payload fields restore (where they are sanitized / allow-listed).
_DEDICATED_KEYS = frozenset(("prompt", "negativePrompt", "imageUrls", "videoUrl", "audioUrl", "startFrame", "endFrame"))


def _is_valid_payload(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and strict_equal(data.get("v"), 1)
        and isinstance(data.get("m"), str)
        and len(data["m"]) > 0
    )


def _filter_urls(urls: Any, warnings: list[str]) -> list[str]:
    result: list[str] = []
    for url in urls:  # `for…of`: a list's items, a string's characters
        if not isinstance(url, str):
            continue
        if is_allowed_url(url):
            result.append(url)
        else:
            warnings.append(f"Rejected URL: {url}")
    return result


def deserialize_payload(encoded: str) -> DeepLinkResult | None:
    """Decode a payload into a :class:`DeepLinkResult`; ``None`` when it does not decode or validate."""
    try:
        data = decode(encoded)
    except Exception:
        return None
    if not _is_valid_payload(data):
        return None

    warnings: list[str] = []
    context: dict[str, Any] = {}
    model = get_model(data["m"])
    if model is None:
        warnings.append(f"Unknown model: {data['m']}")

    if truthy(data.get("p")):
        context["prompt"] = sanitize_prompt(data["p"])
    if truthy(data.get("np")):
        context["negativePrompt"] = sanitize_prompt(data["np"])

    images = data.get("i")
    if isinstance(images, (list, str)) and len(images) > 0:
        valid = _filter_urls(images, warnings)
        if valid:
            context["imageUrls"] = valid
    for field, key, label in _MEDIA:
        value = data.get(field)
        if not truthy(value):
            continue
        if is_allowed_url(value):
            context[key] = value
        else:
            warnings.append(f"{label}: {js_string(value)}")

    # `data.o && typeof data.o === 'object'` — an array is an object too (its indexes become keys).
    params = data.get("o")
    if isinstance(params, list):
        params = {str(i): v for i, v in enumerate(params)}
    if isinstance(params, dict):
        # Deviation from TS (spec §9): TS `Object.assign`s every restored option, so `o.prompt` / `o.startFrame`
        # would overwrite the sanitized prompt and the allow-listed URLs. The serializer never writes these keys
        # to `o`, so SDK-made links decode the same; a crafted link can no longer bypass the checks.
        restored = transfer_values(model.param_config, params) if model is not None else params
        context.update({key: value for key, value in restored.items() if key not in _DEDICATED_KEYS})

    return DeepLinkResult(model_id=data["m"], context=context, model_known=model is not None, warnings=warnings)
