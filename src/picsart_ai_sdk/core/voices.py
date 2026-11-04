"""Default voice ids for the catalog-bound TTS params — port of ``src/core/voices.ts``.

The voice LISTS are not bundled: options come from the platform catalog tasks at runtime
(``ai.catalogs.voices``). ``get_voice_by_id`` lives in :mod:`.catalogs`.
"""

from __future__ import annotations

from .catalogs import get_voice_by_id

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs "George"
GEMINI_DEFAULT_VOICE_ID = "Kore"
DEFAULT_GROK_VOICE_ID = "eve"
ASYNC_DEFAULT_VOICE_ID = "cca0e076-b350-4966-b570-4c2fca50b525"  # "Jennie"
SEEDAUDIO_DEFAULT_VOICE_ID = "en_male_tim_uranus_bigtts"

__all__ = [
    "ASYNC_DEFAULT_VOICE_ID",
    "DEFAULT_GROK_VOICE_ID",
    "DEFAULT_VOICE_ID",
    "GEMINI_DEFAULT_VOICE_ID",
    "SEEDAUDIO_DEFAULT_VOICE_ID",
    "get_voice_by_id",
]
