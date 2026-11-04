import picsart_ai_sdk


def test_exposes_version() -> None:
    assert isinstance(picsart_ai_sdk.__version__, str)
