"""Preset factories for common param descriptors — port of ``src/core/descriptors/presets.ts``.

Each returns a :data:`ModelParams` fragment (one key, two for ``language(True)``) designed to be merged::

    {**p.aspect_ratio(["16:9", "9:16"]), **p.duration([5, 10])}

Internal, as in TS (the package root does not export ``p``). A TS ``undefined`` field is left out of the
descriptor dict, so a descriptor serializes exactly as ``JSON.stringify`` writes the TS one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..catalogs import CatalogSource
from .types import ModelParams, ParamEntry


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _enum(options: list[dict[str, Any]], default: Any, value_type: str) -> dict[str, Any]:
    return {"kind": "enum", "valueType": value_type, "options": options, "default": default}


def _ids(opts: Sequence[Any]) -> list[dict[str, Any]]:
    return [{"id": o} for o in opts]


def _option(opt: Any) -> dict[str, Any]:
    return dict(opt) if isinstance(opt, Mapping) else {"id": opt}


def _first(options: list[dict[str, Any]], default: Any) -> Any:
    return default if default is not None else options[0]["id"]


class _Presets:
    def aspect_ratio(self, opts: Sequence[str] = ("16:9", "9:16", "1:1"), default: str | None = None) -> ModelParams:
        options = _ids(opts)
        return {"aspectRatio": ParamEntry(_enum(options, _first(options, default), "string"))}

    def duration(self, opts: Sequence[int | float | Mapping[str, Any]], default: float | None = None) -> ModelParams:
        """Bare seconds (``[5, 10]``) or ``{"id", "label"}`` pairs — e.g. ``{"id": -1, "label": "Auto"}``."""
        options = [_option(o) for o in opts]
        return {"duration": ParamEntry(_enum(options, _first(options, default), "number"))}

    def duration_range(self, min: float, max: float, default: float, step: float = 1) -> ModelParams:  # noqa: A002
        """Continuous duration (seconds), for a span of more than 10 values (10 or fewer stays an enum)."""
        return {"duration": ParamEntry({"kind": "range", "min": min, "max": max, "step": step, "default": default})}

    def resolution(self, opts: Sequence[str], default: str | None = None) -> ModelParams:
        options = _ids(opts)
        return {"resolution": ParamEntry(_enum(options, _first(options, default), "string"))}

    def count(self, opts: Sequence[int] = (1, 2, 4, 6, 8, 10), default: int | None = None) -> ModelParams:
        options = _ids(opts)
        return {"count": ParamEntry(_enum(options, _first(options, default), "number"))}

    def quality(self, opts: Sequence[str], default: str | None = None) -> ModelParams:
        options = _ids(opts)
        return {"quality": ParamEntry(_enum(options, _first(options, default), "string"))}

    def size(self, opts: Sequence[str], default: str | None = None) -> ModelParams:
        options = _ids(opts)
        return {"size": ParamEntry(_enum(options, _first(options, default), "string"))}

    def style(self, opts: Sequence[Mapping[str, str]], default: str | None = None) -> ModelParams:
        options = [dict(o) for o in opts]
        return {"style": ParamEntry(_enum(options, _first(options, default), "string"))}

    def rendering_speed(self, opts: Sequence[Mapping[str, str]], default: str | None = None) -> ModelParams:
        options = [dict(o) for o in opts]
        return {"renderingSpeed": ParamEntry(_enum(options, _first(options, default), "string"))}

    def generate_audio(self, default: bool = True) -> ModelParams:
        return {"generateAudio": ParamEntry({"kind": "boolean", "default": default})}

    def return_last_frame(self, default: bool = False) -> ModelParams:
        return {"returnLastFrame": ParamEntry({"kind": "boolean", "default": default}, label="Capture Last Frame")}

    def audio_setting(self, opts: Sequence[str] = ("auto", "origin"), default: str = "auto") -> ModelParams:
        """HappyHorse video-edit ``audio_setting``: ``auto`` lets the model decide, ``origin`` keeps the source's."""
        return {"audioSetting": ParamEntry(_enum(_ids(opts), default, "string"), label="Audio")}

    def enhance_prompt(self, default: bool = True) -> ModelParams:
        return {"enhancePrompt": ParamEntry({"kind": "boolean", "default": default})}

    def prompt(
        self,
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        placeholder: str | None = None,
        required: bool | None = None,
    ) -> ModelParams:
        descriptor = _compact(
            {"kind": "text", "minLength": min_length, "maxLength": max_length, "placeholder": placeholder}
        )
        return {"prompt": ParamEntry(descriptor, label="Prompt", required=True if required is None else required)}

    def negative_prompt(self, placeholder: str | None = None, max_length: int | None = None) -> ModelParams:
        descriptor = _compact({"kind": "text", "placeholder": placeholder, "maxLength": max_length})
        return {"negativePrompt": ParamEntry(descriptor, label="Negative Prompt")}

    def seed(self, max: int = 2147483647) -> ModelParams:  # noqa: A002
        """Optional, no default: sent only when the user sets it, so the vendor randomizes otherwise."""
        return {"seed": ParamEntry({"kind": "range", "min": 0, "max": max, "step": 1}, label="Seed")}

    def cfg_scale(self, min: float, max: float, default: float, step: float | None = None) -> ModelParams:  # noqa: A002
        descriptor = _compact({"kind": "range", "min": min, "max": max, "step": step, "default": default})
        return {"cfgScale": ParamEntry(descriptor, label="CFG Scale")}

    def image_weight(self, min: float, max: float, default: float, step: float | None = None) -> ModelParams:  # noqa: A002
        descriptor = _compact({"kind": "range", "min": min, "max": max, "step": step, "default": default})
        return {"imageWeight": ParamEntry(descriptor, label="Image Weight")}

    # ── generic factories (key first) ───────────────────────────────────
    _FILE_BOUNDS = (
        ("max_duration_sec", "maxDurationSec"),
        ("min_duration_sec", "minDurationSec"),
        ("max_frame_rate", "maxFrameRate"),
        ("min_pixels", "minPixels"),
        ("max_pixels", "maxPixels"),
        ("min_side_pixels", "minSidePixels"),
        ("max_side_pixels", "maxSidePixels"),
        ("min_aspect_ratio", "minAspectRatio"),
        ("max_aspect_ratio", "maxAspectRatio"),
        ("max_short_side_pixels", "maxShortSidePixels"),
        ("max_bytes", "maxBytes"),
    )

    def file(
        self,
        key: str,
        accept: str,
        *,
        label: str | None = None,
        required: bool | None = None,
        array: Mapping[str, int] | None = None,
        category: str | None = None,
        **bounds: float,
    ) -> ModelParams:
        """A file slot (``accept``: image / video / audio / media). ``bounds`` are the client-side limits in
        snake_case (``max_duration_sec``, ``max_bytes``, …), written camelCase into the descriptor."""
        known = dict(self._FILE_BOUNDS)
        unknown = set(bounds) - set(known)
        if unknown:
            raise TypeError(f"p.file() got unexpected bounds: {', '.join(sorted(unknown))}")
        descriptor: dict[str, Any] = {"kind": "file", "accept": accept}
        if array is not None:  # JS truthiness: `{}` is truthy, so an empty mapping is kept too
            descriptor["array"] = dict(array)
        for name, wire in self._FILE_BOUNDS:
            if bounds.get(name) is not None:
                descriptor[wire] = bounds[name]
        return {key: ParamEntry(descriptor, label=label, required=required, category=category or None)}  # type: ignore[arg-type]

    def boolean(self, key: str, default: bool, label: str | None = None) -> ModelParams:
        return {key: ParamEntry({"kind": "boolean", "default": default}, label=label)}

    def enum(
        self,
        key: str,
        options: Sequence[str | int | float | Mapping[str, Any]],
        default: Any = None,
        *,
        label: str | None = None,
    ) -> ModelParams:
        """Generic enum select; ``valueType`` is inferred from the first option (number vs string)."""
        normalized = [_option(o) for o in options]
        first = normalized[0]["id"]
        value_type = "number" if isinstance(first, (int, float)) and not isinstance(first, bool) else "string"
        return {key: ParamEntry(_enum(normalized, _first(normalized, default), value_type), label=label)}

    def range(
        self,
        key: str,
        min: float,  # noqa: A002
        max: float,  # noqa: A002
        default: float,
        *,
        step: float | None = None,
        label: str | None = None,
    ) -> ModelParams:
        descriptor = _compact({"kind": "range", "min": min, "max": max, "step": step, "default": default})
        return {key: ParamEntry(descriptor, label=label)}

    def text(
        self,
        key: str,
        *,
        max_length: int | None = None,
        placeholder: str | None = None,
        label: str | None = None,
        required: bool | None = None,
    ) -> ModelParams:
        descriptor = _compact({"kind": "text", "maxLength": max_length, "placeholder": placeholder})
        return {key: ParamEntry(descriptor, label=label, required=required)}

    def catalog(
        self,
        key: str,
        *,
        source: CatalogSource,
        default: str,
        label: str | None = None,
        required: bool | None = None,
    ) -> ModelParams:
        """A param whose options a platform catalog task serves; the value is a free-string id."""
        return {
            key: ParamEntry({"kind": "catalog", "source": source, "default": default}, label=label, required=required)
        }

    def _bound_list(
        self,
        key: str,
        label: str,
        options: Sequence[Mapping[str, str]],
        default: str,
        required: bool | None,
        catalog: CatalogSource | None,
    ) -> ModelParams:
        # Catalog-bound ids are free strings (the live catalog is the source of truth); only unbound lists
        # stay closed enums.
        descriptor = (
            {"kind": "catalog", "source": catalog, "default": default}
            if catalog
            else _enum(
                [{"id": o["id"], "label": o.get("name") if o.get("name") is not None else o["id"]} for o in options],
                default,
                "string",
            )
        )
        return {key: ParamEntry(descriptor, label=label, required=required, catalog_options=[dict(o) for o in options])}

    def voice_id(
        self,
        options: Sequence[Mapping[str, str]],
        default: str,
        *,
        required: bool | None = None,
        catalog: CatalogSource | None = None,
    ) -> ModelParams:
        return self._bound_list("voiceId", "Voice", options, default, required, catalog)

    def video_id(
        self,
        options: Sequence[Mapping[str, str]],
        default: str,
        *,
        required: bool | None = None,
        catalog: CatalogSource | None = None,
    ) -> ModelParams:
        return self._bound_list("videoId", "Avatar", options, default, required, catalog)

    def language(self, has_accent: bool) -> ModelParams:
        out: ModelParams = {"language": ParamEntry({"kind": "text"}, label="Language")}
        if has_accent:
            out["accent"] = ParamEntry({"kind": "text"}, label="Accent")
        return out


p = _Presets()
"""The preset factories (``p.aspect_ratio(...)``, ``p.voice_id(...)``, …)."""

__all__ = ["p"]
