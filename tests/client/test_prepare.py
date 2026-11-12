from typing import Any

import pytest

from picsart_ai_sdk.client.prepare import (
    MODELS_WORKFLOW,
    build_models_command,
    parse_dynamic_result,
    parse_dynamic_run_result,
    parse_dynamic_text_result,
)
from picsart_ai_sdk.client.types import GenerateResultItem
from picsart_ai_sdk.core.errors import ApiError
from picsart_ai_sdk.core.js import js_string, truthy
from picsart_ai_sdk.core.response import is_seedance_draft_task, throw_if_error_result, to_completed_status
from picsart_ai_sdk.core.workflow import WorkflowJobHandle


def done(result: Any, task_id: str = "task-1", raw: Any = None, usage: Any = None) -> Any:
    return to_completed_status(
        WorkflowJobHandle(MODELS_WORKFLOW, task_id), result, raw if raw is not None else result, usage
    )


def test_command_shapes() -> None:
    assert build_models_command("m", {"prompt": "p"}) == {"model": "m", "params": {"prompt": "p"}}
    assert build_models_command("m", {}, with_options=True) == {
        "model": "m",
        "params": {},
        "options": {"inputs_transformation": {"downscale_oversized_images": False}},
    }
    assert build_models_command("m", {}, with_options=True, drive={"name": "x"}, downscale_oversized_images=True)[
        "options"
    ] == {"inputs_transformation": {"downscale_oversized_images": True}, "drive": {"name": "x"}}


def test_command_service_tier() -> None:
    # The only option of a text / run / price command …
    assert build_models_command("m", {}, service_tier="priority") == {
        "model": "m",
        "params": {},
        "options": {"service_tier": "priority"},
    }
    # … and beside the media options of a generation.
    assert build_models_command("m", {}, with_options=True, service_tier="auto")["options"] == {
        "inputs_transformation": {"downscale_oversized_images": False},
        "service_tier": "auto",
    }


def test_items_with_metadata_and_usage() -> None:
    res = parse_dynamic_result(
        done({"items": [{"url": "a", "metadata": {"exploreImageId": "e"}}, {"url": "b"}]}, usage={"credits": 4}), "m"
    )
    assert res.url == "a" and res.items == res.results
    assert res.items == [GenerateResultItem("a", {"exploreImageId": "e"}), GenerateResultItem("b")]
    assert (res.generation_id, res.usage) == ("task-1", {"credits": 4})


def test_url_only_and_sync_without_id() -> None:
    res = parse_dynamic_result(done({"url": "u"}, task_id=""), "m")
    assert res.items == [GenerateResultItem("u")] and res.generation_id is None


@pytest.mark.parametrize("result", [{}, {"items": []}, {"items": [{"metadata": {}}]}, None, ["u"]])
def test_no_url_is_invalid_response(result: Any) -> None:
    with pytest.raises(ApiError) as exc:
        parse_dynamic_result(done(result), "flux")
    assert (exc.value.message, exc.value.status, exc.value.code) == (
        "flux: unexpected response — no result URL",
        502,
        "invalid_response",
    )


def test_text_result() -> None:
    res = parse_dynamic_text_result(done({"text": "hi", "model": "x"}, raw={"text": "hi", "model": "x"}), "claude")
    assert (res.text, res.model) == ("hi", "x")
    assert parse_dynamic_text_result(done({"text": ""}), "claude").model == "claude"
    with pytest.raises(ApiError, match="claude: unexpected response — no text"):
        parse_dynamic_text_result(done({"text": 1}), "claude")


def test_run_result() -> None:
    res = parse_dynamic_run_result(done({"a": 1}), "m")
    assert (res.result, res.status, res.id) == ({"a": 1}, "COMPLETED", "task-1")
    assert parse_dynamic_run_result(done(0, task_id=""), "m").id is None  # 0 is a result, not "empty"
    with pytest.raises(ApiError, match="m: unexpected response — empty result"):
        parse_dynamic_run_result(to_completed_status(WorkflowJobHandle("v1/models", "t"), None, None), "m")


@pytest.mark.parametrize(
    ("payload", "message", "status", "code"),
    [
        ({"status": 500, "message": "boom"}, "flux failed (500): boom", 500, "server_error"),
        ({"status": "FAILED", "reason": "nsfw"}, "flux failed: nsfw", 502, "nsfw"),
        ({"statusCode": 429}, "flux failed (429): unknown error", 429, "rate_limited"),
        ({"status": "error", "error": "bad"}, "flux failed: bad", 502, "bad_gateway"),
        ({"status": 422, "message": "", "reason": "x"}, "flux failed (422): unknown error", 422, "x"),
    ],
)
def test_throw_if_error_result(payload: dict[str, Any], message: str, status: int, code: str) -> None:
    with pytest.raises(ApiError) as exc:
        throw_if_error_result(payload, "flux")
    assert (exc.value.message, exc.value.status, exc.value.code) == (message, status, code)


@pytest.mark.parametrize(
    "payload", [None, [{"status": 500}], "x", {"status": 200}, {"status": True}, {"status": "COMPLETED"}]
)
def test_not_an_error(payload: Any) -> None:
    throw_if_error_result(payload, "m")


def test_failed_result_fails_text_parse_through_raw() -> None:
    with pytest.raises(ApiError, match="m failed: x"):
        parse_dynamic_text_result(done({"text": "t"}, raw={"status": "FAILED", "message": "x"}), "m")


def test_js_helpers() -> None:
    values: list[object] = [[], {}, "", 0, 0.0, None, False, "a", 1]
    assert [truthy(v) for v in values] == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    strings: list[object] = [5.0, 2.5, True, None, [1, None], {"a": 1}]
    assert [js_string(v) for v in strings] == [
        "5",
        "2.5",
        "true",
        "null",
        "1,",
        "[object Object]",
    ]
    assert is_seedance_draft_task({"id": "d"}) and not is_seedance_draft_task({"id": 1})
