"""Live e2e helpers — port of ``__tests__/e2e/helpers/ai-sdk-test-client.ts``.

Needs ``PICSART_TOKEN``; ``PICSART_API_URL`` defaults to the production gateway.
Skipped (not failed) without a token, so the unit run stays offline.
"""

import os
from typing import Any

import httpx
import pytest

from picsart_ai_sdk import AiClient, FetchInit, create_client
from picsart_ai_sdk.client.fetch import send

TOKEN = os.environ.get("PICSART_TOKEN")
API_URL = os.environ.get("PICSART_API_URL", "https://api.picsart.com")

requires_token = pytest.mark.skipif(not TOKEN, reason="PICSART_TOKEN is not set")


@pytest.fixture
async def ai() -> Any:
    http = httpx.AsyncClient(timeout=None)

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        headers = {
            **(init.get("headers") or {}),
            "content-type": "application/json",
            "authorization": f"Bearer {TOKEN}",
            "country-code": "US",
            "platform": "test",
            "x-touchpoint": "ai-sdk.e2e",
        }
        return await send(http, url, {**init, "headers": headers})

    client: AiClient = create_client(api_url=API_URL, fetch=fetch)
    yield client
    await http.aclose()
