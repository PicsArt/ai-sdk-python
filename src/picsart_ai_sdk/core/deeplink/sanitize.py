"""URL allow-list and prompt sanitizing — port of ``src/core/deeplink/sanitize.ts``.

The JS regexes are reproduced exactly: ``$`` is end of input (``\\Z``), ``/i`` folds ASCII only, ``\\w`` is
``[A-Za-z0-9_]`` and ``\\s`` is the JS whitespace set.
"""

from __future__ import annotations

import re

from ..js import JS_WHITESPACE, js_string

_FLAGS = re.IGNORECASE | re.ASCII
_ALLOWED_URL_PATTERNS = [
    re.compile(r"^https://([a-z0-9-]+\.)*picsart\.com(/|\Z)", _FLAGS),
    re.compile(r"^https://([a-z0-9-]+\.)*pastatic\.com(/|\Z)", _FLAGS),
    re.compile(r"^https://([a-z0-9-]+\.)*cdn-picsart\.com(/|\Z)", _FLAGS),
]
_JS_S = "[" + re.escape(JS_WHITESPACE) + "]"
_TAG = re.compile(r"</?[^>]+(>|\Z)")
_JS_PROTOCOL = re.compile(r"javascript" + _JS_S + "*:", _FLAGS)
_WORD = re.compile(r"[A-Za-z0-9_]+")
_JS_WHITESPACE = frozenset(JS_WHITESPACE)


def _strip_handlers(text: str) -> str:
    """``text.replace(/on\\w+\\s*=/gi, '')`` in linear time.

    Python's backtracking ``re`` takes quadratic time on that regex over a long word with no ``=`` after it
    (``"onon…on"``). ``\\w``, ``\\s`` and ``=`` are disjoint, so a match is: the leftmost ``on`` of a maximal word
    run that has at least one word character after it, through the run, the whitespace after it and an ``=``.
    """
    out: list[str] = []
    last = 0
    for run in _WORD.finditer(text):
        start, end = run.span()
        j = end
        while j < len(text) and text[j] in _JS_WHITESPACE:
            j += 1
        if j >= len(text) or text[j] != "=":
            continue
        at = run.group().lower().find("on", 0, end - start - 1)  # `on` + at least one more word char
        if at < 0:
            continue
        out.append(text[last : start + at])
        last = j + 1
    out.append(text[last:])
    return "".join(out)


def is_allowed_url(url: object) -> bool:
    """Whether ``url`` is an https URL on a Picsart CDN domain.

    TS also requires ``new URL(url)`` to parse with protocol ``https:``. Every string the patterns accept
    parses, except a host with an invalid punycode label (``xn--a.picsart.com``), which TS rejects and this
    accepts — still a Picsart domain, listed in spec §9.
    """
    text = js_string(url)
    return any(pattern.search(text) for pattern in _ALLOWED_URL_PATTERNS)


def sanitize_prompt(text: str) -> str:
    """Strip HTML/script tags, ``javascript:`` and inline ``on…=`` handlers from prompt text."""
    if not isinstance(text, str):
        raise TypeError("text.replace is not a function")  # what TS throws for a non-string prompt
    return _strip_handlers(_JS_PROTOCOL.sub("", _TAG.sub("", text)))
