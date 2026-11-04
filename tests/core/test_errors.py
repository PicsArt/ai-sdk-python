import pytest

from picsart_ai_sdk.core.errors import ApiError, code_for_status


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "bad_request"),
        (401, "unauthorized"),
        (402, "payment_required"),
        (403, "forbidden"),
        (404, "not_found"),
        (408, "timeout"),
        (409, "conflict"),
        (413, "payload_too_large"),
        (422, "unprocessable_entity"),
        (429, "rate_limited"),
        (500, "server_error"),
        (502, "bad_gateway"),
        (503, "service_unavailable"),
        (504, "gateway_timeout"),
        (507, "server_error"),
        (418, "http_418"),
    ],
)
def test_code_for_status(status: int, code: str) -> None:
    assert code_for_status(status) == code


def test_api_error_fields() -> None:
    e = ApiError("boom", status=502, code="generation_failed")
    assert (str(e), e.message, e.status, e.code, e.reason, e.name) == (
        "boom",
        "boom",
        502,
        "generation_failed",
        "generation_failed",
        "ApiError",
    )
    assert isinstance(e, Exception)
