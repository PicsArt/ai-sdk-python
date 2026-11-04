"""Stage 2 final-review fixes: shared loads survive a cancelled caller, pricing as the pricing SDK, odd providers."""

import asyncio
from typing import Any

import httpx
import pytest

from picsart_ai_sdk import PricingOptions, catalog, create_client
from picsart_ai_sdk.client.fetch import FetchInit
from picsart_ai_sdk.core.active_catalog import _reset_active_catalog_for_tests, active_models
from picsart_ai_sdk.core.catalog_store import _reset_model_catalog_store_for_tests, load_model_catalog
from picsart_ai_sdk.core.catalog_wire import deserialize_catalog, serialize_catalog
from picsart_ai_sdk.core.descriptors.pricing import (
    ModelPricingClientError,
    ModelPricingServerError,
    ModelPricingUnknownError,
    _reset_pricing_for_tests,
)
from tests.fixtures.catalog import FIXTURE_CATALOG

FIXTURE = serialize_catalog(FIXTURE_CATALOG)


@pytest.fixture(autouse=True)
def reset() -> Any:
    for f in (_reset_model_catalog_store_for_tests, _reset_active_catalog_for_tests, _reset_pricing_for_tests):
        f()
    yield
    for f in (_reset_model_catalog_store_for_tests, _reset_active_catalog_for_tests, _reset_pricing_for_tests):
        f()


def gated(body: Any) -> tuple[asyncio.Event, Any]:
    gate = asyncio.Event()

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        await gate.wait()
        return httpx.Response(200, json=body)

    return gate, fetch


# ── C1: one cancelled waiter never cancels the shared load ─────────────
async def test_cancelling_one_catalog_waiter_keeps_the_shared_load() -> None:
    gate, fetch = gated(FIXTURE)

    async def wait_load() -> None:
        await load_model_catalog(fetch, "https://x.test")

    a, b = asyncio.ensure_future(wait_load()), asyncio.ensure_future(wait_load())
    await asyncio.sleep(0)
    a.cancel()
    await asyncio.sleep(0)
    gate.set()
    await b
    assert a.cancelled() and len(active_models()) == len(FIXTURE_CATALOG)


async def test_validate_timeout_keeps_the_shared_load() -> None:
    gate, fetch = gated(FIXTURE)
    ai = create_client(api_url="https://x.test", fetch=fetch)
    loading = asyncio.ensure_future(ai.init_model_catalog())
    await asyncio.sleep(0)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(ai.validate(FIXTURE_CATALOG[0].id, {}), 0.01)
    gate.set()
    await loading
    assert len(active_models()) == len(FIXTURE_CATALOG)


async def test_cancelling_one_pricing_waiter_keeps_the_shared_load() -> None:
    gate, fetch = gated({"status": "success", "response": []})
    catalog.pricing.configure(PricingOptions(base_url="https://p.test", fetch=fetch))

    async def wait_load() -> None:
        await catalog.pricing.load()

    a, b = asyncio.ensure_future(wait_load()), asyncio.ensure_future(wait_load())
    await asyncio.sleep(0)
    a.cancel()
    await asyncio.sleep(0)
    gate.set()
    await b
    assert catalog.pricing.is_loaded()


# ── I1: a user-discounts failure never fails the pricing load ──────────
async def test_user_discounts_failure_is_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        if url.endswith("/user-discounts/public"):
            return httpx.Response(503, json={"message": "down"})
        return httpx.Response(200, json={"status": "success", "response": []})

    catalog.pricing.configure(PricingOptions(base_url="https://p.test", fetch=fetch))
    await catalog.pricing.load()
    assert catalog.pricing.is_loaded()
    assert "ModelPricingClient: failed to load user discounts - ModelPricingServerError: " in caplog.text


# ── I2: pricing errors as the pricing SDK words them ───────────────────
def pricing_answering(status: int, body: Any = None, content: bytes | None = None) -> None:
    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        if url.endswith("/user-discounts/public"):
            return httpx.Response(200, json={"status": "success", "response": []})
        if content is not None:
            return httpx.Response(status, content=content)
        return httpx.Response(status, json=body)

    catalog.pricing.configure(PricingOptions(base_url="https://p.test", fetch=fetch))


async def test_pricing_client_error_message_and_details() -> None:
    pricing_answering(404, {"reason": "not_found", "message": "nope"})
    with pytest.raises(ModelPricingClientError) as exc:
        await catalog.pricing.load()
    assert str(exc.value) == "ModelPricingClientError: [404] getModelPricing failed: not_found - nope"
    assert exc.value.status == 404 and exc.value.details == {"reason": "not_found", "message": "nope"}
    pricing_answering(400, {})
    with pytest.raises(
        ModelPricingClientError, match=r"^ModelPricingClientError: \[400\] getModelPricing failed:  - $"
    ):
        await catalog.pricing.load()


async def test_pricing_server_error_message() -> None:
    pricing_answering(500, {"reason": "boom"})
    with pytest.raises(ModelPricingServerError) as exc:
        await catalog.pricing.load()
    assert str(exc.value) == "ModelPricingServerError: [500] - getModelPricing failed with message: boom."


async def test_pricing_unknown_error_wraps_other_failures() -> None:
    pricing_answering(404, content=b"<html>")
    with pytest.raises(
        ModelPricingUnknownError, match=r"^ModelPricingUnknownError: modelPricing\.getModelPricing failed - "
    ):
        await catalog.pricing.load()

    async def broken(url: str, init: FetchInit) -> httpx.Response:
        raise httpx.ConnectError("refused")

    catalog.pricing.configure(PricingOptions(base_url="https://p.test", fetch=broken))
    with pytest.raises(ModelPricingUnknownError, match=r"modelPricing\.getModelPricing failed - refused$"):
        await catalog.pricing.load()


# ── I3: PricingOptions.headers reach every request ─────────────────────
async def test_pricing_headers_are_merged_into_requests() -> None:
    seen: list[dict[str, str]] = []

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        seen.append(dict(init.get("headers") or {}))
        return httpx.Response(200, json={"status": "success", "response": []})

    catalog.pricing.configure(PricingOptions(base_url="https://p.test", fetch=fetch, headers={"X-App": "editor"}))
    await catalog.pricing.load()
    assert seen == [{"accept": "application/json", "x-app": "editor"}] * 2


# ── I4: provider checked with JS truthiness, non-objects tolerated ─────
@pytest.mark.parametrize("provider", [{}, "openai", 1, []])
def test_odd_provider_values_load_with_no_provider(provider: Any) -> None:
    wire = serialize_catalog(FIXTURE_CATALOG[:1])
    wire["response"]["models"][0]["provider"] = provider
    [model] = deserialize_catalog(wire)
    assert model.id == FIXTURE_CATALOG[0].id and model.provider is None and model.provider_name is None


@pytest.mark.parametrize("provider", [None, "", 0])
def test_falsy_provider_is_still_invalid(provider: Any) -> None:
    wire = serialize_catalog(FIXTURE_CATALOG[:1])
    wire["response"]["models"][0]["provider"] = provider
    with pytest.raises(Exception, match="missing `id` or `provider`"):
        deserialize_catalog(wire)


# ── a superseded pricing load never publishes ─────────────────────────
async def test_reconfigured_pricing_ignores_the_superseded_load() -> None:
    old_gate = asyncio.Event()

    def pricing_fetch(credits: int, gate: asyncio.Event | None) -> Any:
        async def fetch(url: str, init: FetchInit) -> httpx.Response:
            if gate is not None:
                await gate.wait()
            if "user-discounts" in url:
                return httpx.Response(200, json={"response": []})
            entry = {"credits": credits, "unit": "image", "metadata": {"modelId": "m1"}}
            return httpx.Response(200, json={"response": [entry]})

        return fetch

    catalog.pricing.configure(PricingOptions(base_url="https://old.test", fetch=pricing_fetch(1, old_gate)))
    old = asyncio.ensure_future(catalog.pricing.load())
    await asyncio.sleep(0)
    catalog.pricing.configure(PricingOptions(base_url="https://new.test", fetch=pricing_fetch(9, None)))
    await catalog.pricing.load()
    old_gate.set()
    await old
    from picsart_ai_sdk.core.descriptors.pricing import get_credits_for_model

    credits = get_credits_for_model("m1")
    assert credits is not None and (credits.min, credits.max) == (9, 9)
