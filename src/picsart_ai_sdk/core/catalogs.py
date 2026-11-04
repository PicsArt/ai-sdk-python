"""Runtime param catalogs (voices, avatars, templates) — port of ``src/core/catalogs.ts``.

Shapes off the wire keep their TS keys; the hydration registry is process-global
as in TS (a hydrated catalog feeds ``Model(id).params().param(key)["catalogOptions"]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from typing_extensions import NotRequired

from .types import AvatarOption, VoiceOption


class CatalogPreview(TypedDict, total=False):
    imageUrl: str
    videoUrl: str
    audioUrl: str


class CatalogItem(TypedDict):
    id: str
    name: str
    description: NotRequired[str]
    tags: list[str]
    preview: NotRequired[CatalogPreview]
    meta: NotRequired[dict[str, Any]]


class CatalogQuery(TypedDict, total=False):
    modelId: str
    cursor: str
    limit: int


class CatalogResult(TypedDict):
    items: list[CatalogItem]
    version: str
    ttlSeconds: float
    nextCursor: str | None


class CatalogSource(TypedDict):
    workflow: str
    modelId: NotRequired[str]


def _meta_string(item: CatalogItem, key: str) -> str | None:
    value = (item.get("meta") or {}).get(key)
    return value if isinstance(value, str) else None


def _without_none(d: dict[str, Any]) -> dict[str, Any]:
    # JS keeps `key: undefined`, which JSON drops; a missing key reads the same through `.get`.
    return {k: v for k, v in d.items() if v is not None}


def to_voice_option(item: CatalogItem, provider: str) -> VoiceOption:
    preview = item.get("preview") or {}
    return _without_none(
        {  # type: ignore[return-value]
            "id": item["id"],
            "name": item["name"],
            "description": item.get("description") if item.get("description") is not None else "",
            "tags": item["tags"],
            "provider": provider,
            "previewUrl": preview.get("audioUrl"),
        }
    )


def to_avatar_option(item: CatalogItem, provider: str) -> AvatarOption:
    preview = item.get("preview") or {}
    return _without_none(
        {  # type: ignore[return-value]
            "id": item["id"],
            "name": item["name"],
            "description": item.get("description") if item.get("description") is not None else "",
            "tags": item["tags"],
            "provider": provider,
            "previewImageUrl": preview.get("imageUrl"),
            "previewVideoUrl": preview.get("videoUrl"),
            "gender": _meta_string(item, "gender"),
            "defaultVoiceId": _meta_string(item, "defaultVoiceId"),
        }
    )


@dataclass
class HydratedCatalog:
    param_key: str
    items: list[CatalogItem]
    options: list[dict[str, str]] = field(default_factory=list)
    catalog_options: list[Any] = field(default_factory=list)
    version: str = ""


_registry: dict[str, HydratedCatalog] = {}


def _key_of(source: CatalogSource) -> str:
    return f"{source['workflow']} {source.get('modelId') or ''}"


_OPTION_ADAPTERS: dict[str, Any] = {"voiceId": to_voice_option, "videoId": to_avatar_option}


def install_hydrated_catalog(
    source: CatalogSource, param_key: str, items: list[CatalogItem], provider: str, version: str
) -> None:
    adapt = _OPTION_ADAPTERS.get(param_key)
    _registry[_key_of(source)] = HydratedCatalog(
        param_key=param_key,
        items=items,
        options=[{"id": i["id"], "label": i["name"]} for i in items],
        catalog_options=[adapt(i, provider) for i in items] if adapt else list(items),
        version=version,
    )


def get_hydrated_catalog(source: CatalogSource) -> HydratedCatalog | None:
    return _registry.get(_key_of(source))


def get_hydrated_voices() -> list[VoiceOption]:
    out: list[VoiceOption] = []
    for c in _registry.values():
        if c.param_key == "voiceId":
            out.extend(c.catalog_options)
    return out


def get_voice_by_id(voice_id: str) -> VoiceOption | None:
    """A voice from the hydrated voice catalogs (``ai.catalogs.voices()`` hydrates them)."""
    return next((v for v in get_hydrated_voices() if v["id"] == voice_id), None)


def clear_hydrated_catalogs() -> None:
    _registry.clear()
