"""Model pricing for ``catalog.pricing`` / ``get_credits_info`` — port of ``src/core/descriptors/pricing.ts``
and of the part of ``@picsart/pa-model-pricing-sdk`` it uses (load model pricing + public user discounts).

The pricing table is grouped by ``metadata.modelId`` once, on the first load (as in TS); credit overrides
and user discounts never apply here because no country / touchpoint / user context is passed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from ...client.fetch import AuthenticatedFetch, FetchInit, send
from ..js import js_string, truthy
from .types import CreditRange, CreditRangeContext, CreditTier

DEFAULT_TIMEOUT_MS = 5000
log = logging.getLogger("picsart_ai_sdk")


class ModelPricingServerError(Exception):
    """A 5xx from the pricing service."""

    def __init__(self, message: str) -> None:
        super().__init__(f"ModelPricingServerError: {message}")


class ModelPricingClientError(Exception):
    """A non-OK, non-5xx answer from the pricing service; ``details`` is its JSON body."""

    def __init__(self, action: str, status: int, body: Any) -> None:
        reason = body.get("reason") if isinstance(body, dict) else None
        message = body.get("message") if isinstance(body, dict) else None
        super().__init__(
            f"ModelPricingClientError: [{status}] {action} failed: "
            f"{js_string(reason) if truthy(reason) else ''} - {js_string(message) if truthy(message) else ''}"
        )
        self.status, self.details = status, body


class ModelPricingUnknownError(Exception):
    """Any other pricing failure (network, a non-JSON body, …)."""

    def __init__(self, message: str) -> None:
        super().__init__(f"ModelPricingUnknownError: {message}")


@dataclass(frozen=True)
class PricingOptions:
    """``ModelPricingClientOptions``: the pricing service base URL, extra request headers and an optional
    authenticated fetch."""

    base_url: str
    fetch: AuthenticatedFetch | None = None
    headers: Mapping[str, str] | None = None
    timeout_ms: float = DEFAULT_TIMEOUT_MS


_options: PricingOptions | None = None
_by_model: dict[str, list[dict[str, Any]]] | None = None
_load: asyncio.Future[None] | None = None


def configure_pricing(options: PricingOptions) -> None:
    global _options, _by_model, _load
    _options, _by_model, _load = options, None, None


def _request_headers(options: PricingOptions) -> dict[str, str]:
    """``new Headers({ Accept, ...options.headers })`` as the SDK hands it to a custom fetch: lower-cased, sorted."""
    merged: dict[str, str] = {}
    for key, value in {"Accept": "application/json", **(options.headers or {})}.items():
        merged[key.lower()] = value
    return dict(sorted(merged.items()))


async def _get(options: PricingOptions, path: str, action: str) -> Any:
    url = f"{options.base_url.rstrip('/')}{path}"
    init: FetchInit = {"method": "GET", "headers": _request_headers(options)}
    timeout = options.timeout_ms / 1000
    try:
        if options.fetch is not None:
            res = await asyncio.wait_for(options.fetch(url, init), timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as http:
                res = await send(http, url, init)
                await res.aread()
        if res.status_code >= 500:
            try:
                body = json.loads(await res.aread())
                message = (
                    (body.get("message") or body.get("reason") or "Unknown error")
                    if isinstance(body, dict)
                    else "Unknown error"
                )
            except ValueError:
                message = "Non json response was returned from server"
            raise ModelPricingServerError(f"[{res.status_code}] - {action} failed with message: {message}.")
        if not res.is_success:
            raise ModelPricingClientError(action, res.status_code, json.loads(await res.aread()))
        return json.loads(await res.aread())
    except (
        ModelPricingClientError,
        ModelPricingServerError,
        TimeoutError,
        asyncio.TimeoutError,
        asyncio.CancelledError,
    ):
        raise  # TS passes its own errors and the timeout DOMException through unwrapped
    except Exception as error:
        raise ModelPricingUnknownError(f"modelPricing.{action} failed - {error}") from error


async def _load_user_discounts(options: PricingOptions) -> None:
    """``loadUserDiscounts``: never fails the load — a failure is logged and discounts stay empty."""
    try:
        await _get(options, "/pricing-management/user-discounts/public", "getUserDiscounts")
    except Exception as error:
        log.error("ModelPricingClient: failed to load user discounts - %s", error)


def load_pricing() -> Awaitable[None]:
    """Load the pricing table once (``catalog.pricing.load()``); every caller gets a shielded view of it."""
    global _load
    loop = asyncio.get_running_loop()
    if _by_model is not None:
        done: asyncio.Future[None] = loop.create_future()
        done.set_result(None)
        return done
    if _options is None:
        failed: asyncio.Future[None] = loop.create_future()
        failed.set_exception(
            Exception("loadPricing(): not configured. Call catalog.pricing.configure({ baseUrl, fetch }) first.")
        )
        return failed
    if _load is None:
        options = _options

        async def run() -> None:
            global _by_model, _load
            try:
                pricing, _ = await asyncio.gather(
                    _get(options, "/pricing-management/model-pricing", "getModelPricing"),
                    _load_user_discounts(options),
                )
            except BaseException:
                if _load is task:  # a superseded load must not clear the current one
                    _load = None
                raise
            by_model: dict[str, list[dict[str, Any]]] = {}
            for entry in (pricing or {}).get("response") or []:
                by_model.setdefault(entry["metadata"]["modelId"], []).append(entry)
            if _load is task:  # configure_pricing() since this load started: its table is obsolete
                _by_model = by_model

        task: asyncio.Future[None] = asyncio.ensure_future(run())
        _load = task
    return asyncio.shield(_load)


def is_pricing_loaded() -> bool:
    return _by_model is not None


def get_credits_for_model(model_id: str, ctx: CreditRangeContext | None = None) -> CreditRange | None:
    if _by_model is None:
        return None
    entries = _by_model.get(model_id) or []
    if not entries:
        return None
    if ctx is not None:
        entries = [
            e
            for e in entries
            if (ctx.generate_audio is None or e["metadata"].get("audio") is ctx.generate_audio)
            and (ctx.resolution is None or e["metadata"].get("quality") == ctx.resolution)
        ]
        if not entries:
            return None
    lo, hi = math.inf, -math.inf
    unit: str | None = entries[0].get("unit")
    for e in entries:
        lo, hi = min(lo, e["credits"]), max(hi, e["credits"])
        if e.get("unit") != unit:
            unit = None
    tiers = [
        CreditTier(
            credits=e["credits"],
            unit=e.get("unit"),  # type: ignore[arg-type]
            quality=e["metadata"].get("quality") or None,
            audio=e["metadata"].get("audio"),
            use_case=e["metadata"].get("useCase"),
        )
        for e in entries
    ]
    return CreditRange(min=lo, max=hi, tiers=tiers, unit=unit or None)


def _reset_pricing_for_tests() -> None:
    global _options, _by_model, _load
    _options, _by_model, _load = None, None, None
