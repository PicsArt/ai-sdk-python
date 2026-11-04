import asyncio

import pytest

from picsart_ai_sdk.abort import AbortController, AbortError, link_signals


def test_abort_sets_state_and_keeps_first_reason() -> None:
    c = AbortController()
    assert not c.signal.aborted and c.signal.reason is None
    c.abort("first")
    c.abort("second")
    assert c.signal.aborted and c.signal.reason == "first"


def test_default_reason_is_an_abort_error() -> None:
    c = AbortController()
    c.abort()
    assert isinstance(c.signal.reason, AbortError)


def test_listener_fires_once_and_not_after_removal() -> None:
    c = AbortController()
    calls: list[str] = []
    c.signal.add_listener(lambda: calls.append("a"))
    remove = c.signal.add_listener(lambda: calls.append("b"))
    remove()
    c.abort()
    c.abort()
    assert calls == ["a"]


def test_listener_on_already_aborted_signal_fires_immediately() -> None:
    c = AbortController()
    c.abort()
    calls: list[int] = []
    c.signal.add_listener(lambda: calls.append(1))
    assert calls == [1]


def test_link_signals() -> None:
    parent, child = AbortController(), AbortController()
    link_signals(parent.signal, child)
    parent.abort("p")
    assert child.signal.aborted and child.signal.reason == "p"

    done, late = AbortController(), AbortController()
    done.abort()
    link_signals(done.signal, late)
    assert late.signal.aborted

    other = AbortController()
    unlink = link_signals(None, other)
    unlink()
    assert not other.signal.aborted


def test_unlink_stops_propagation() -> None:
    parent, child = AbortController(), AbortController()
    unlink = link_signals(parent.signal, child)
    unlink()
    parent.abort()
    assert not child.signal.aborted


def test_throw_if_aborted() -> None:
    c = AbortController()
    c.signal.throw_if_aborted()
    c.abort()
    with pytest.raises(AbortError):
        c.signal.throw_if_aborted()


async def test_wait_returns_after_abort() -> None:
    c = AbortController()
    task = asyncio.create_task(c.signal.wait())
    await asyncio.sleep(0)
    assert not task.done()
    c.abort()
    await asyncio.wait_for(task, 1)
