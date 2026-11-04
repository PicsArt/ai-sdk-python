"""Port of ``__tests__/unit/active-catalog.test.ts`` (registry part; Model/catalog accessors in their own suite)."""

import pytest

from picsart_ai_sdk.core.active_catalog import (
    _reset_active_catalog_for_tests,
    active_models,
    install_catalog,
    is_catalog_installed,
    peek_active_models,
)
from picsart_ai_sdk.core.errors import ApiError
from picsart_ai_sdk.core.model_registry import find_model, get_model, resolve_model
from picsart_ai_sdk.core.visibility import is_visible_for_releases, release_of
from tests.fixtures.catalog import FIXTURE_CATALOG, FLUX, RICH, make_model


@pytest.fixture(autouse=True)
def reset() -> None:
    _reset_active_catalog_for_tests()


def test_pending_by_default() -> None:
    assert not is_catalog_installed() and peek_active_models() is None
    for call in (active_models, lambda: resolve_model("flux-2-pro"), lambda: get_model("x")):
        with pytest.raises(ApiError) as exc:
            call()
        assert (exc.value.status, exc.value.code) == (409, "catalog_not_loaded")
    assert str(exc.value) == "Model catalog not loaded — call ai.initModelCatalog() before using the model catalog."


def test_install_reflects_everywhere_and_reinstall_swaps() -> None:
    install_catalog(FIXTURE_CATALOG)
    assert is_catalog_installed() and active_models() is FIXTURE_CATALOG and peek_active_models() is FIXTURE_CATALOG
    assert resolve_model("flux-2-pro").name == "Flux 2 Pro"
    assert find_model("kling/v1/video").id == "kling-v3"  # type: ignore[union-attr]
    assert find_model("  Flux 2 Pro ").id == "flux-2-pro"  # type: ignore[union-attr]
    install_catalog([FLUX])
    assert find_model("kling-v3") is None


def test_lookup_order_and_ambiguous_names() -> None:
    install_catalog(
        [RICH, make_model(id="a", name="Same", mode="image"), make_model(id="b", name="same ", mode="image")]
    )
    assert get_model("rich-vendor-id") is RICH
    assert find_model("fixture/v1/rich-edit") is RICH
    assert find_model("same") is None and find_model("") is None
    with pytest.raises(ApiError) as exc:
        resolve_model("nope")
    assert (exc.value.message, exc.value.status, exc.value.code) == ('Unknown model: "nope"', 400, "unknown_model")


def test_visibility() -> None:
    assert release_of(make_model(id="x", name="x", mode="image", release=None)) == "production"
    assert not is_visible_for_releases(make_model(id="p", name="p", mode="image", release="preview"))
    assert is_visible_for_releases(make_model(id="p", name="p", mode="image", release="preview"), ["preview"])
    assert not is_visible_for_releases(make_model(id="d", name="d", mode="image", deprecated=True), ["production"])
