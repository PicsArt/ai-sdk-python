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


@requires_token
async def test_basic_generate(ai: Any) -> None:
    from picsart_ai_sdk import Model

    await ai.init_model_catalog()
    params = (
        json.loads(MODEL_PARAMS)
        if MODEL_PARAMS
        else {
            **Model(MODEL_ID).params().get_defaults(),
            **({"prompt": "a red apple on a white table"} if Model(MODEL_ID).params().has_param("prompt") else {}),
        }
    )
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
