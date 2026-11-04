"""JavaScript semantics the TS SDK relies on, for byte-identical behavior."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any


def truthy(value: object) -> bool:
    """JS truthiness: only ``None``, ``False``, ``0``, ``NaN`` and ``""`` are falsy (``[]`` / ``{}`` are truthy)."""
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not (value == 0 or (isinstance(value, float) and math.isnan(value)))
    if isinstance(value, str):
        return value != ""
    return True


def is_number(value: object) -> bool:
    """``typeof value === 'number'`` (booleans are not numbers)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def nullish(*values: Any) -> Any:
    """``a ?? b ?? c``: the first value that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def js_string(value: object) -> str:
    """``String(value)``."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return js_number(value)
    if isinstance(value, list):
        return ",".join("" if v is None else js_string(v) for v in value)
    if isinstance(value, dict):
        return "[object Object]"
    return str(value)


def _utf16(s: str) -> bytes:
    """``s`` as UTF-16 code units; a lone surrogate (valid in a JS string) is one unit, as in JS."""
    return s.encode("utf-16-le", errors="surrogatepass")


def utf16_len(s: str) -> int:
    """``s.length`` in JavaScript: UTF-16 code units."""
    return len(_utf16(s)) // 2


def utf16_slice(s: str, end: int) -> str:
    """``s.slice(0, end)`` in UTF-16 code units (a cut surrogate pair leaves its high surrogate, as in JS)."""
    return _utf16(s)[: max(0, end) * 2].decode("utf-16-le", errors="surrogatepass")


def utf16_code_at(s: str, index: int) -> int:
    """``s.charCodeAt(index)``."""
    units = _utf16(s)
    return int.from_bytes(units[index * 2 : index * 2 + 2], "little")


def js_number(value: float) -> str:
    """``Number.prototype.toString()``: shortest round-trip digits, ECMAScript notation thresholds."""
    if math.isnan(value) or math.isinf(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    mantissa, _, exp_part = repr(abs(value)).partition("e")
    int_part, _, frac_part = mantissa.partition(".")
    digits = (int_part + frac_part).lstrip("0")
    leading = len(int_part + frac_part) - len((int_part + frac_part).lstrip("0"))
    n = len(int_part) + (int(exp_part) if exp_part else 0) - leading  # decimal point position
    digits = digits.rstrip("0") or "0"
    k = len(digits)
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        exp = f"e{'+' if e >= 0 else '-'}{abs(e)}"
        out = (digits if k == 1 else f"{digits[0]}.{digits[1:]}") + exp
    return sign + out


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _js_json_string(s: str) -> str:
    """A JSON string as well-formed ``JSON.stringify`` writes it: a surrogate pair split into two code points
    is one character (as in JS), and a lone surrogate is escaped as ``\\udXXX`` instead of failing to encode."""
    joined = s.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
    return _LONE_SURROGATE.sub(lambda m: f"\\u{ord(m.group()):04x}", json.dumps(joined, ensure_ascii=False))


def js_json_dumps(value: Any) -> str:
    """``JSON.stringify(value)``: compact, non-ASCII kept, JS number notation, NaN/Infinity as ``null``."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "null"
        return js_number(value)
    if isinstance(value, str):
        return _js_json_string(value)
    if isinstance(value, dict):
        return "{" + ",".join(f"{_js_json_string(str(k))}:{js_json_dumps(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(js_json_dumps(v) for v in value) + "]"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def strict_equal(a: object, b: object) -> bool:
    """``a === b``: same JS type and value (``True`` ≠ ``1``); objects and arrays by identity."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if is_number(a) and is_number(b):
        return a == b  # NaN !== NaN, 1 === 1.0
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is b
    return a is b


def same_value_zero(a: object, b: object) -> bool:
    """``Array.prototype.includes`` equality: ``===`` except ``NaN`` equals ``NaN``."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return strict_equal(a, b)


def js_includes(items: list[Any], value: object) -> bool:
    return any(same_value_zero(item, value) for item in items)


JS_WHITESPACE = (
    " \t\n\r\x0b\x0c\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)
"""What JS ``\\s`` and ``trim()`` treat as whitespace (WhiteSpace + LineTerminator)."""


def js_trim(s: str) -> str:
    """``String.prototype.trim()`` (JS whitespace and line terminators)."""
    return s.strip(JS_WHITESPACE)


_JS_DECIMAL = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)
_JS_RADIX = {"0x": 16, "0o": 8, "0b": 2}


def _number(value: float) -> float:
    """A JS number as Python: integral values within 2**53 as ``int``."""
    return int(value) if math.isfinite(value) and value.is_integer() and abs(value) <= 2**53 else value


def js_to_number(value: object) -> float:
    """``Number(value)``. ``None`` reads as ``null`` (0); a string goes through JS numeric-literal rules."""
    if value is None or value is False:
        return 0
    if value is True:
        return 1
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return js_to_number(js_string(list(value))) if len(value) <= 1 else math.nan
    if not isinstance(value, str):
        return math.nan
    text = value.strip(JS_WHITESPACE)
    if text == "":
        return 0
    if text in ("Infinity", "+Infinity"):
        return math.inf
    if text == "-Infinity":
        return -math.inf
    radix = _JS_RADIX.get(text[:2].lower())
    if radix is not None:
        digits = text[2:]
        valid = {16: "0123456789abcdefABCDEF", 8: "01234567", 2: "01"}[radix]
        return int(digits, radix) if digits and all(c in valid for c in digits) else math.nan
    if not _JS_DECIMAL.fullmatch(text):
        return math.nan
    return _number(float(text))


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Unexpected token {name} in JSON")


def js_json_parse(text: str | bytes) -> Any:
    """``JSON.parse(text)`` (no NaN / Infinity literals; raises ``ValueError`` like a ``SyntaxError``)."""
    return json.loads(text, parse_constant=_reject_constant)


def js_prop(obj: object, key: str) -> Any:
    """``obj.key``: ``TypeError`` on ``null`` / ``undefined`` (``None``) as in JS, ``None`` for a missing key or a
    primitive, the value otherwise."""
    if obj is None:
        raise TypeError(f"Cannot read properties of null (reading '{key}')")
    return obj.get(key) if isinstance(obj, Mapping) else None
