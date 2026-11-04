"""Size limits — port of ``src/core/limits.ts`` (lengths in UTF-16 code units, as in JS)."""

MAX_DRIVE_PROMPT_LENGTH = 18_000
"""The prompt stored with a Drive generation is cut to this."""

MAX_DRIVE_ATTRIBUTE_LENGTH = 20_000
"""A Drive attribute value (the serialized ``aiSDKPayload``) must not exceed this."""
