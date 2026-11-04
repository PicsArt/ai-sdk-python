"""Ports of ``model-accessor.test.ts`` and ``validate.test.ts``, plus pricing and catalog-aware naming."""

import asyncio
import json
import re
from typing import Any

import httpx
import pytest

from picsart_ai_sdk import (
    ApiError,
    CreditRangeContext,
    DriveConfig,
    GenerateOptions,
    Model,
    PricingOptions,
    ValidationResult,
    catalog,
    create_client,
)
from picsart_ai_sdk.client import workflows
from picsart_ai_sdk.client.fetch import FetchInit
from picsart_ai_sdk.core.active_catalog import _reset_active_catalog_for_tests, install_catalog
from picsart_ai_sdk.core.catalog_store import _reset_model_catalog_store_for_tests
from picsart_ai_sdk.core.catalog_wire import serialize_catalog
from picsart_ai_sdk.core.catalogs import clear_hydrated_catalogs, install_hydrated_catalog
from picsart_ai_sdk.core.descriptors.pricing import _reset_pricing_for_tests
from picsart_ai_sdk.core.descriptors.types import ParamEntry
from tests.fixtures.catalog import FIXTURE_CATALOG, KLING, RICH, make_model


@pytest.fixture(autouse=True)
def reset() -> Any:
    for f in (
        _reset_active_catalog_for_tests,
        _reset_model_catalog_store_for_tests,
        _reset_pricing_for_tests,
        clear_hydrated_catalogs,
    ):
        f()
    yield
    for f in (
        _reset_active_catalog_for_tests,
        _reset_model_catalog_store_for_tests,
        _reset_pricing_for_tests,
        clear_hydrated_catalogs,
    ):
        f()


def test_model_needs_a_catalog_and_a_known_id() -> None:
    with pytest.raises(ApiError) as exc:
        Model("flux-2-pro")
    assert exc.value.code == "catalog_not_loaded"
    install_catalog(FIXTURE_CATALOG)
    with pytest.raises(ApiError) as exc:
        Model("nope")
    assert (exc.value.status, exc.value.code) == (400, "unknown_model")


def test_accessor_mechanics() -> None:
    install_catalog([RICH])
    m = Model("rich-fixture")
    assert (m.id, m.name, m.api.workflow, m.api.edit_workflow) == (
        "rich-fixture",
        "Rich Fixture",
        "fixture/v1/rich-fixture",
        "fixture/v1/rich-edit",
    )
    p = m.params()
    assert p is m.params() and m.meta() is m.meta()  # memoized
    assert p.param("prompt") == {"label": "Prompt", "required": True, "kind": "text", "maxLength": 2000}
    assert p.has_param("steps") and not p.has_param("nope") and p.param("nope") is None
    assert [e["key"] for e in p.all()][:3] == ["prompt", "aspectRatio", "voiceId"]
    assert p.enum("aspectRatio")["default"] == "1:1" and p.enum("steps") is None  # type: ignore[index]
    assert p.range("steps")["max"] == 50 and p.boolean("hd")["default"] is False  # type: ignore[index]
    assert p.aspect_ratio() is not None and p.prompt() is not None and p.duration() is None
    assert p.has_file_input() and p.get_default("steps") == 20 and p.get_default("prompt") is None
    assert p.get_defaults() == {"aspectRatio": "1:1", "voiceId": "", "steps": 20, "hd": False}
    assert p.to_schema()["steps"] == {"type": "number", "min": 1, "max": 50, "default": 20}
    assert p.transfer_values({"aspectRatio": "16:9", "steps": 99, "gone": 1}) == {
        "aspectRatio": "16:9",
        "voiceId": "",
        "steps": 50,
        "hd": False,
    }
    meta = m.meta()
    assert (meta.mode, meta.input_type, meta.badges, meta.release, meta.added_at, meta.provider.label) == (
        "image",
        "t2i",
        ["new", "popular"],
        "production",
        None,
        "FX",
    )


def test_hydrated_catalog_options() -> None:
    install_catalog([RICH])
    assert "catalogOptions" not in Model("rich-fixture").params().param("voiceId")  # type: ignore[operator]
    install_hydrated_catalog(
        {"workflow": "fixture/v1/catalog/voices"},
        "voiceId",
        [{"id": "v1", "name": "Ann", "tags": [], "preview": {"audioUrl": "a.mp3"}}],
        "flux",
        "1",
    )
    options = Model("rich-fixture").params().param("voiceId")["catalogOptions"]  # type: ignore[index]
    assert options == [
        {"id": "v1", "name": "Ann", "description": "", "tags": [], "provider": "flux", "previewUrl": "a.mp3"}
    ]


def test_params_for_applies_constraints() -> None:
    model = make_model(
        id="c",
        name="C",
        mode="video",
        param_config={
            "resolution": ParamEntry(
                {"kind": "enum", "valueType": "string", "options": [{"id": "720p"}, {"id": "1080p"}], "default": "720p"}
            ),
            "duration": ParamEntry(
                {"kind": "enum", "valueType": "number", "options": [{"id": 5}, {"id": 10}], "default": 5}
            ),
            "audio": ParamEntry({"kind": "boolean", "default": True}),
            "prompt": ParamEntry({"kind": "text"}),
        },
        constraints=[
            {
                "when": {"resolution": {"is": "1080p"}},
                "then": {
                    "duration": {"allowed": [5], "reason": "1080p is 5s"},
                    "audio": {"disabled": True, "reason": "no audio"},
                },
            },
            {"when": {"resolution": {"is": "1080p"}}, "then": {"prompt": {"required": True, "reason": "describe it"}}},
        ],
    )
    install_catalog([model])
    m = Model("c")
    assert m.params_for({"resolution": "720p"}) is m.params()
    p = m.params_for({"resolution": "1080p"})
    assert p.enum("duration")["options"] == [{"id": 5}, {"id": 10, "disabled": True, "disabledReason": "1080p is 5s"}]  # type: ignore[index]
    assert (p.boolean("audio")["disabled"], p.boolean("audio")["disabledReason"]) == (True, "no audio")  # type: ignore[index]
    assert (p.text("prompt")["required"], p.text("prompt")["requiredReason"]) == (True, "describe it")  # type: ignore[index]
    assert {e["key"]: e.get("disabled") for e in p.all()}["audio"] is True
    assert m.validate({"resolution": "1080p"}) == ValidationResult(False, ['"prompt" is required: describe it'])


def test_validate_results() -> None:
    clean = make_model(id="clean", name="Clean", mode="image", param_config={**RICH.param_config})
    install_catalog([clean, RICH])
    m = Model("clean")
    assert m.validate({"prompt": "a cat", "turns": [{"text": "hi"}]}) == ValidationResult(valid=True)
    assert m.validate({}) == ValidationResult(False, ['"prompt" is required'])
    assert m.validate(["x"]) == ValidationResult(False, ['Invalid input for model "clean"'])
    assert m.validate({"prompt": "x", "steps": True}).errors == ['"steps" must be a number']
    # RICH's fixture constraint has no operator (`when: {resolution: "1080p"}`): TS throws a TypeError
    # inside validate() and reports it; the port does the same.
    assert Model("rich-fixture").validate({"prompt": "a cat"}).errors == [
        "Cannot use 'in' operator to search for 'exists' in 1080p"
    ]


def test_catalog_all_find_search() -> None:
    preview = make_model(id="prev", name="Preview Model", mode="image", release="preview")
    install_catalog([*FIXTURE_CATALOG, preview])
    assert [m.id for m in catalog.all()] == ["flux-2-pro", "claude-opus-5", "kling-v3"]
    assert [m.id for m in catalog.all(release=["preview"])] == ["prev"]
    assert [m.id for m in catalog.find(output="video")] == ["kling-v3"]
    assert [m.id for m in catalog.find(provider="anthropic")] == ["claude-opus-5"]
    assert [m.id for m in catalog.search("FLUX")] == ["flux-2-pro"]
    assert [m.id for m in catalog.search("kling")] == ["kling-v3"]


async def test_client_validate_before_during_and_after_load() -> None:
    gate = asyncio.Event()

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        await gate.wait()
        models = serialize_catalog(
            [
                make_model(
                    id="flux-2-pro",
                    name="Flux 2 Pro",
                    mode="image",
                    param_config={
                        "prompt": ParamEntry({"kind": "text"}, required=True),
                        "backendOnlyParam": ParamEntry({"kind": "text"}, required=True),
                    },
                )
            ]
        )
        return httpx.Response(200, json=models)

    ai = create_client(api_url="https://api.test", fetch=fetch)
    with pytest.raises(ApiError) as exc:
        await ai.validate("flux-2-pro", {"prompt": "x"})
    assert (exc.value.status, exc.value.code, exc.value.message) == (
        409,
        "catalog_not_loaded",
        "Model catalog not loaded — call ai.initModelCatalog() before ai.validate().",
    )
    loading = asyncio.ensure_future(ai.init_model_catalog())
    await asyncio.sleep(0)
    pending = asyncio.ensure_future(ai.validate("flux-2-pro", {"prompt": "x"}))
    await asyncio.sleep(0)
    assert not pending.done()
    gate.set()
    await loading
    assert (await pending).errors == ['"backendOnlyParam" is required']
    with pytest.raises(ApiError) as exc:
        await ai.validate("nope", {})
    assert exc.value.code == "unknown_model"


async def test_pricing_credit_info() -> None:
    pricing = [
        {
            "metadata": {"modelId": "kling-v3", "quality": "720p", "audio": False, "useCase": "t2v"},
            "credits": 10,
            "unit": "second",
        },
        {
            "metadata": {"modelId": "kling-v3", "quality": "1080p", "audio": True, "useCase": "t2v"},
            "credits": 20,
            "unit": "second",
        },
        {"metadata": {"modelId": "rich-vendor-id", "quality": "", "audio": False}, "credits": 3, "unit": "generation"},
    ]
    seen: list[str] = []

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        seen.append(url)
        if url.endswith("/model-pricing"):
            return httpx.Response(200, json={"status": "success", "response": pricing})
        return httpx.Response(200, json={"status": "success", "response": []})

    install_catalog([KLING, RICH])
    with pytest.raises(Exception, match=r"loadPricing\(\): not configured"):
        await catalog.pricing.load()
    assert not catalog.pricing.is_loaded() and Model("kling-v3").get_credits_info() is None
    catalog.pricing.configure(PricingOptions(base_url="https://pricing.test/", fetch=fetch))
    await catalog.pricing.load()
    await catalog.pricing.load()
    assert catalog.pricing.is_loaded() and len(seen) == 2
    info = Model("kling-v3").get_credits_info()
    assert (info.min, info.max, info.unit, len(info.tiers)) == (10, 20, "second", 2)  # type: ignore[union-attr]
    one = Model("kling-v3").get_credits_info(CreditRangeContext(resolution="1080p", generate_audio=True))
    assert (one.min, one.max, one.tiers[0].quality, one.tiers[0].use_case) == (20, 20, "1080p", "t2v")  # type: ignore[union-attr]
    assert Model("kling-v3").get_credits_info(CreditRangeContext(resolution="4k")) is None
    rich = Model("rich-fixture").get_credits_info()  # by vendor modelId first
    assert (rich.min, rich.tiers[0].quality) == (3, None)  # type: ignore[union-attr]


async def test_generate_names_files_from_the_installed_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fast(_: float) -> None:
        return None

    monkeypatch.setattr(workflows, "_sleep", fast)
    submits: list[dict[str, Any]] = []

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        if "models-catalog" in url:
            return httpx.Response(200, json=serialize_catalog(FIXTURE_CATALOG))
        if url.endswith("/submit"):
            submits.append(json.loads(init["body"])["params"])
            return httpx.Response(200, json={"status": "success", "response": {"id": "t"}})
        return httpx.Response(
            200, json={"status": "success", "response": {"status": "COMPLETED", "result": {"url": "https://c/x"}}}
        )

    ai = create_client(api_url="https://api.test", fetch=fetch, drive=DriveConfig(folder="AI"))
    await ai.init_model_catalog()
    await ai.generate("Flux 2 Pro", {"prompt": "a dog"}, GenerateOptions(interval_ms=1))
    assert submits[-1]["model"] == "Flux 2 Pro"  # the ref is sent verbatim
    assert re.fullmatch(r"a-dog-\d+\.png", submits[-1]["options"]["drive"]["name"])
    assert submits[-1]["options"]["drive"]["attributes"]["model"] == "flux-2-pro"
    await ai.generate("kling-v3", {"prompt": "a cat"}, GenerateOptions(interval_ms=1))
    assert re.fullmatch(r"a-cat-\d+\.mp4", submits[-1]["options"]["drive"]["name"])
