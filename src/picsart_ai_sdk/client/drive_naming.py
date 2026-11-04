"""Drive file naming and generation provenance used by ``generate`` — port of the naming and
attribute helpers of ``src/client/drive.ts`` (the Drive client itself comes with ``ai.drive``)."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict
from urllib.parse import urlsplit

from ..core.js import js_json_dumps, js_string, nullish, utf16_code_at, utf16_len, utf16_slice
from ..core.limits import MAX_DRIVE_ATTRIBUTE_LENGTH, MAX_DRIVE_PROMPT_LENGTH

AppType = Literal["native", "miniapp"]


@dataclass(frozen=True)
class AppIdentity:
    id: str
    type: AppType


@dataclass(frozen=True)
class GenerationProvenance:
    model_id: str
    params: dict[str, Any]
    app: AppIdentity | None = None


class FilenameHints(TypedDict, total=False):
    mimeType: str
    url: str
    format: str


class ModelLike(Protocol):
    """What naming needs from a model definition (the catalog's, in step 2)."""

    @property
    def param_config(self) -> Mapping[str, Any]: ...

    @property
    def output_extension(self) -> str | None: ...


_MIME_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/pjpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/avif": "avif",
    "image/heic": "heic",
    "image/heif": "heif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/svg+xml": "svg",
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/webm": "webm",
    "video/x-matroska": "mkv",
    "video/x-msvideo": "avi",
    "video/x-m4v": "m4v",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/vnd.wave": "wav",
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/webm": "webm",
}

_EXTENSION_ALIASES = {"jpeg": "jpg", "jpe": "jpg", "tif": "tiff", "mpeg4": "mp4", "quicktime": "mov"}

_FORMAT_PARAM_KEYS = ("outputFormat", "output_format", "format")


def infer_resource_type(mode: str) -> Literal["PHOTO", "VIDEO", "AUDIO"]:
    if mode == "video":
        return "VIDEO"
    if mode == "audio":
        return "AUDIO"
    return "PHOTO"


def _normalize_extension(raw: str | None) -> str | None:
    if not raw:
        return None
    ext = re.sub(r"^\.", "", raw.strip().lower()).split("_")[0]
    if not re.fullmatch(r"[a-z0-9]{2,5}", ext):
        return None
    return _EXTENSION_ALIASES.get(ext, ext)


def _extension_from_mime(mime_type: str | None) -> str | None:
    if not mime_type:
        return None
    return _MIME_EXTENSIONS.get(mime_type.split(";")[0].strip().lower())


def _extension_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parts = urlsplit(url)
    # `new URL()` throws without a scheme and host; TS then splits the raw string.
    path = parts.path if parts.scheme and parts.netloc else re.split(r"[?#]", url)[0]
    match = re.search(r"\.([A-Za-z0-9]{2,5})$", path.split("/")[-1])
    return _normalize_extension(match.group(1) if match else None)


def _default_extension(mode: str) -> str:
    return {"video": "mp4", "audio": "mp3", "image": "png"}.get(mode, "")


def resolve_extension(mode: str, hints: FilenameHints | None = None) -> str:
    """MIME type → URL path extension → requested format → mode default (``""`` for unknown modes)."""
    h = hints or {}
    found = nullish(
        _extension_from_mime(h.get("mimeType")),
        _extension_from_url(h.get("url")),
        _normalize_extension(h.get("format")),
    )
    return found if isinstance(found, str) else _default_extension(mode)


def _descriptor_default(entry: object) -> object:
    descriptor = entry.get("descriptor") if isinstance(entry, Mapping) else getattr(entry, "descriptor", None)
    if isinstance(descriptor, Mapping):
        return descriptor.get("default")
    return getattr(descriptor, "default", None)


def expected_output_format(model: ModelLike, params: Mapping[str, Any]) -> str | None:
    """The explicit format param, else its catalog default, else the model's declared extension."""
    for key in _FORMAT_PARAM_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    config = model.param_config or {}
    for key in _FORMAT_PARAM_KEYS:
        fallback = _descriptor_default(config.get(key))
        if isinstance(fallback, str) and fallback:
            return fallback
    return model.output_extension


def build_filename(prompt: str | None, mode: str, hints: FilenameHints | None = None) -> str:
    short_id = str(int(time.time() * 1000))[-6:]
    ext = resolve_extension(mode, hints)
    suffix = f".{ext}" if ext else ""
    if not prompt:
        return f"ai-generation-{short_id}{suffix}"
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())
    slug = re.sub(r"^-|-$", "", slug)[:50]
    return f"{slug}-{short_id}{suffix}"


def _cut_prompt(prompt: str, maximum: int) -> str:
    if utf16_len(prompt) <= maximum:
        return prompt
    code = utf16_code_at(prompt, maximum - 1)
    return utf16_slice(prompt, maximum - 1 if 0xD800 <= code <= 0xDBFF else maximum)


def cut_stored_prompt(prompt: str) -> str:
    """A prompt cut to what Drive stores with a generation (18 000 code units, no split surrogate pair)."""
    return _cut_prompt(prompt, MAX_DRIVE_PROMPT_LENGTH)


def to_sdk_payload(params: Mapping[str, Any]) -> dict[str, Any]:
    """``{prompt}`` (cut to 18 000 code units) then every other non-empty param, in order."""
    payload: dict[str, Any] = {"prompt": cut_stored_prompt(js_string(nullish(params.get("prompt"), "")))}
    for key, value in params.items():
        if key == "prompt" or value is None or value == "" and isinstance(value, str):
            continue
        payload[key] = value
    return payload


def _serialize_sdk_payload(params: Mapping[str, Any]) -> str:
    payload = to_sdk_payload(params)
    prompt: str = payload["prompt"]
    while True:
        serialized = js_json_dumps({**payload, "prompt": prompt})
        excess = utf16_len(serialized) - MAX_DRIVE_ATTRIBUTE_LENGTH
        if excess <= 0 or not prompt:
            return serialized
        prompt = _cut_prompt(prompt, max(0, utf16_len(prompt) - excess))


def build_generation_attributes(provenance: GenerationProvenance) -> dict[str, str]:
    """``{model, aiSDKPayload, appId?, appType?}`` — what a Drive file records about its generation."""
    attrs = {"model": provenance.model_id, "aiSDKPayload": _serialize_sdk_payload(provenance.params)}
    if provenance.app is not None:
        attrs["appId"] = provenance.app.id
        attrs["appType"] = provenance.app.type
    return attrs
