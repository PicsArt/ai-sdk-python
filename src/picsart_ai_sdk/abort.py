"""Cancellation, as in the TypeScript SDK (``signal?: AbortSignal``).

A small port of the DOM ``AbortController`` / ``AbortSignal`` the TS SDK takes,
so the same calls cancel the same way and raise the same 499 ``aborted``
errors. Native ``asyncio`` task cancellation works too and propagates as
:class:`asyncio.CancelledError`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class AbortError(Exception):
    """Raised when an operation is aborted through an :class:`AbortSignal` (the DOM ``AbortError``)."""

    name = "AbortError"

    def __init__(self, message: str = "This operation was aborted") -> None:
        super().__init__(message)


class AbortSignal:
    """Read side of an :class:`AbortController`."""

    def __init__(self) -> None:
        self._aborted = False
        self._reason: object | None = None
        self._listeners: list[Callable[[], None]] = []
        self._event: asyncio.Event | None = None

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> object | None:
        return self._reason

    def add_listener(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Call ``callback`` once on abort (at once if already aborted). Returns a remover."""
        if self._aborted:
            callback()
            return lambda: None
        self._listeners.append(callback)

        def remove() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return remove

    def throw_if_aborted(self) -> None:
        if self._aborted:
            raise self._reason if isinstance(self._reason, BaseException) else AbortError()

    async def wait(self) -> None:
        """Return once the signal is aborted."""
        if self._aborted:
            return
        if self._event is None:
            self._event = asyncio.Event()
        await self._event.wait()

    def _abort(self, reason: object | None) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._reason = AbortError() if reason is None else reason
        if self._event is not None:
            self._event.set()
        listeners, self._listeners = self._listeners, []
        for listener in listeners:
            listener()


class AbortController:
    """Owns an :class:`AbortSignal`; :meth:`abort` aborts it (idempotent, the first reason wins)."""

    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: object | None = None) -> None:
        self.signal._abort(reason)


def link_signals(parent: AbortSignal | None, child: AbortController) -> Callable[[], None]:
    """Abort ``child`` when ``parent`` aborts (now, if it already has). Returns an unlinker."""
    if parent is None:
        return lambda: None
    return parent.add_listener(lambda: child.abort(parent.reason))
