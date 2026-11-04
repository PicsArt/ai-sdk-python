"""Port of ``__tests__/unit/catalog-wire.test.ts``."""

import copy
from typing import Any

import pytest

from picsart_ai_sdk.core.catalog_wire import deserialize_catalog, serialize_catalog
from picsart_ai_sdk.core.errors import ApiError
from tests.fixtures.catalog import FIXTURE_CATALOG, RICH, serializable_view

MODELS = [*FIXTURE_CATALOG, RICH]


def test_round_trip_every_fixture_model() -> None:
    back = deserialize_catalog(serialize_catalog(MODELS))
    assert [serializable_view(m) for m in back] == [serializable_view(m) for m in MODELS]


def test_provider_block_flattens() -> None:
    wire = serialize_catalog([RICH])
    assert wire["response"]["models"][0]["provider"] == {
        "id": "flux",
        "name": "Fixture",
        "color": "#123456",
        "label": "FX",
    }
    [back] = deserialize_catalog(wire)
    assert (back.provider, back.provider_name) == (RICH.provider, RICH.provider_name)


def test_badges_and_kinds_and_bounds() -> None:
    wire = serialize_catalog([RICH])
    assert wire["response"]["models"][0]["badges"] == ["new", "popular"]
    kinds = {e.descriptor["kind"] for e in RICH.param_config.values()}
    assert {"enum", "catalog", "range", "boolean", "text", "file", "object"} <= kinds
    [back] = deserialize_catalog(wire)
    assert back.badge == RICH.badge
    assert back.param_config["turns"] == RICH.param_config["turns"]
    for bound in ("minPixels", "minSidePixels", "maxShortSidePixels", "maxDurationSec", "maxBytes"):
        assert back.param_config["image"].descriptor[bound] == RICH.param_config["image"].descriptor[bound]
    assert back.constraints == RICH.constraints


def test_routing_less_model_gets_empty_workflow() -> None:
    wire = copy.deepcopy(serialize_catalog([RICH]))
    for key in ("workflow", "editWorkflow", "modelId"):
        del wire["response"]["models"][0][key]
    [back] = deserialize_catalog(wire)
    assert back.workflow == "" and back.edit_workflow is None


def test_entry_meta_is_lifted_out_of_the_param() -> None:
    wire = serialize_catalog([RICH])
    assert wire["response"]["models"][0]["params"]["prompt"] == {
        "kind": "text",
        "maxLength": 2000,
        "label": "Prompt",
        "required": True,
    }


@pytest.mark.parametrize("bad", [None, {}, {"status": "error"}, {"status": "success", "response": {}}])
def test_malformed_envelope(bad: Any) -> None:
    with pytest.raises(ApiError) as exc:
        deserialize_catalog(bad)
    assert (exc.value.status, exc.value.code) == (502, "invalid_response")


def test_model_without_id_or_provider() -> None:
    with pytest.raises(ApiError, match="model entry is missing `id` or `provider`"):
        deserialize_catalog({"status": "success", "response": {"models": [{"id": "x"}]}})
