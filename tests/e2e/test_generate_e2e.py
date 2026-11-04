"""Live smoke — port of ``__tests__/e2e/generate.e2e.ts`` and ``generate-text.e2e.ts``.

``generate`` builds its input from the catalog defaults of ``TEST_MODEL_ID`` (TS
``catalog-loader.buildContext``); ``TEST_MODEL_PARAMS`` (JSON) overrides it.
"""

import json
import os
import re
from typing import Any

import pytest

from .conftest import requires_token

# Permanent CMS-uploaded test assets the backend can access (TS catalog-loader TEST_ASSETS).
TEST_ASSETS = {
    "image": "https://cdn-cms-uploads.picsart.com/cms-uploads/61beebb2-1028-408f-970f-1e43e9ce538d.png",
    "video": "https://cdn-cms-uploads.picsart.com/cms-uploads/5aa76de2-4847-4126-9c1e-5677846fab1d.mp4",
    "audio": "https://cdn-cms-uploads.picsart.com/cms-uploads/3dbb86fc-8625-4481-a1af-6870fe678f5a.mp3",
    "media": "https://cdn-cms-uploads.picsart.com/cms-uploads/61beebb2-1028-408f-970f-1e43e9ce538d.png",
}
PROMPT = "A calm ocean at sunset"

MODEL_ID = os.environ.get("TEST_MODEL_ID", "picsart-sod-v8-2")
MODEL_PARAMS = os.environ.get("TEST_MODEL_PARAMS")

TEXT_MODELS = [
    "claude-opus-4-8",
    "gpt-5.5",
    "gemini-3-pro",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "kimi-k3",
    "grok-4.7",
    "minimax-m3",
    "minimax-m3.1-flash",
]


def _asset(accept: Any) -> str:
    return TEST_ASSETS.get(accept, TEST_ASSETS["image"])


def _object_value(d: dict[str, Any]) -> Any:
    if not d.get("array"):
        return _object_item(d.get("fields") or {})
    count = max(1, (d["array"] or {}).get("min") or 1) if isinstance(d["array"], dict) else 1
    return [_object_item(d.get("fields") or {}) for _ in range(count)]


def _object_item(fields: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key, field in fields.items():
        kind = field.get("kind")
        if kind == "text":
            item[key] = field.get("placeholder") or "sample text"
        elif kind == "file":
            item[key] = _asset(field.get("accept"))
        elif kind == "object":
            item[key] = _object_value(field)
        elif field.get("default") is not None:
            item[key] = field["default"]
    return item


def build_default_context(model_id: str) -> dict[str, Any]:
    """Port of TS ``catalog-loader.buildDefaultContext``: defaults, plus samples for required inputs."""
    from picsart_ai_sdk import Model

    ctx: dict[str, Any] = {"prompt": PROMPT}
    for entry in Model(model_id).params().all():
        key, kind, required = entry["key"], entry.get("kind"), entry.get("required")
        if kind in ("enum", "catalog", "range", "boolean"):
            ctx[key] = entry.get("default")
        elif kind == "text":
            if key == "prompt":
                ctx[key] = PROMPT
            elif required:
                ctx[key] = "es" if key == "language" else "sample text"
        elif kind == "file" and required:
            url = _asset(entry.get("accept"))
            ctx[key] = [url] if entry.get("array") else url
        elif kind == "object" and required:
            ctx[key] = _object_value(entry)
    return ctx


@requires_token
async def test_basic_generate(ai: Any) -> None:
    await ai.init_model_catalog()
    params = json.loads(MODEL_PARAMS) if MODEL_PARAMS else build_default_context(MODEL_ID)
    result = await ai.generate(MODEL_ID, params)
    assert len(result.items) >= 1
    assert result.url == result.items[0].url
    assert re.match(r"^https?://", result.items[0].url)


@requires_token
@pytest.mark.parametrize("model", TEXT_MODELS)
async def test_generate_text_pong(ai: Any, model: str) -> None:
    res = await ai.generate_text(model, {"prompt": "Reply with exactly the word: pong"})
    assert res.model
    assert res.text.strip()
    assert res.raw is not None
