"""Provider display metadata — port of ``src/core/providers.ts`` (internal; the catalog carries the
same fields per model, so this table is only a fallback)."""

from __future__ import annotations

from typing import TypedDict


class ProviderMeta(TypedDict):
    color: str
    label: str
    name: str


def _p(color: str, label: str, name: str) -> ProviderMeta:
    return {"color": color, "label": label, "name": name}


PROVIDERS: dict[str, ProviderMeta] = {
    "picsart": _p("#FF3399", "PA", "Picsart"),
    "google": _p("#4285F4", "G", "Google"),
    "kling": _p("#8B5CF6", "K", "Kling"),
    "grok": _p("#1DA1F2", "X", "Grok"),
    "openai": _p("#10B981", "O", "OpenAI"),
    "flux": _p("#FF6B6B", "F", "Flux"),
    "ideogram": _p("#06B6D4", "I", "Ideogram"),
    "elevenlabs": _p("#2D6B4F", "XI", "ElevenLabs"),
    "minimax": _p("#FF7B54", "M", "MiniMax"),
    "wan": _p("#00BCD4", "W", "Wan"),
    "seedance": _p("#EC4899", "SD", "Seedance"),
    "ltx": _p("#6366F1", "LT", "LTX"),
    "seedream": _p("#14B8A6", "SR", "Seedream"),
    "seedaudio": _p("#7C3AED", "SA", "Seed Audio"),
    "hunyuan": _p("#F59E0B", "HY", "Hunyuan"),
    "pika": _p("#FF6B9D", "PK", "Pika"),
    "runway": _p("#00D4AA", "RW", "Runway"),
    "luma": _p("#FFB800", "LU", "Luma"),
    "ovi": _p("#9333EA", "OV", "OVI"),
    "creatify": _p("#FF3D00", "CR", "Creatify"),
    "veed": _p("#5B21B6", "VD", "VEED"),
    "bytedance": _p("#00F5D4", "BD", "ByteDance"),
    "qwen": _p("#1E40AF", "QW", "Qwen"),
    "reve": _p("#E11D48", "RV", "Reve"),
    "recraft": _p("#3B82F6", "RF", "Recraft"),
    "videography": _p("#78716C", "VG", "Videography"),
    "topaz": _p("#14B8A6", "TZ", "Topaz"),
    "heygen": _p("#5B4EFF", "HG", "HeyGen"),
    "happyhorse": _p("#FF6A00", "HH", "Happy Horse"),
    "pixverse": _p("#7C3AED", "PV", "PixVerse"),
    "anthropic": _p("#D97757", "CL", "Anthropic"),
    "async": _p("#5E5CE6", "AA", "Async AI"),
    "captionsai": _p("#1D1F20", "MR", "Mirage"),
    "meta": _p("#0081FB", "MT", "Meta"),
    "typesafe": _p("#3E6B8F", "TS", "TypeSafe"),
    "depth-anything": _p("#0EA5E9", "DA", "Depth Anything"),
    "kimi": _p("#1A1A2E", "KM", "Kimi"),
    "mirelo": _p("#5C3DF5", "ML", "Mirelo"),
}


def get_provider_color(provider: str) -> str:
    meta = PROVIDERS.get(provider)
    return meta["color"] if meta is not None else "#666"


def get_provider_label(provider: str) -> str:
    meta = PROVIDERS.get(provider)
    return meta["label"] if meta is not None else "?"
