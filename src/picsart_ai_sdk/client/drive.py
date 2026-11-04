"""``ai.drive`` — folders, listing, saving and reactions in Picsart Drive — port of ``src/client/drive.ts``.

Uses the same authenticated fetch as the generation client. The root folder is found or created on first
use and cached. Every operation swallows a failure and answers ``None`` / ``[]`` / ``False``, as in TS;
where TS lets an error through (a malformed file in a listing), so does this port.

File naming and generation attributes live in :mod:`.drive_naming` (shared with the generation surface).
"""

from __future__ import annotations

import asyncio
import builtins
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

import httpx

from ..core.js import js_json_dumps, js_json_parse, js_prop, js_string, js_to_number, js_trim, truthy
from ..core.response import is_seedance_draft_task
from .drive_naming import (
    AppType,
    GenerationProvenance,
    build_filename,
    build_generation_attributes,
    cut_stored_prompt,
    infer_resource_type,
)
from .fetch import AuthenticatedFetch, FetchInit

MediaTypeFilter = Literal["image", "video", "audio"]
UserReaction = Literal["like", "dislike"]
ResourceType = Literal["PHOTO", "VIDEO", "AUDIO"]

USER_REACTION_ATTR = "userReaction"
# Attributes the seedance worker stamps on a saved Seedance 2.5 draft: its signed reference (JSON) and when
# the vendor drops it (ISO time). Namespaced like every custom Drive attribute (`namespace~name`).
DRAFT_TASK_ATTR = "seedance~draftTask"
DRAFT_EXPIRES_AT_ATTR = "seedance~draftExpiresAt"

# ── result types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class DriveFolder:
    """A folder in Picsart Drive."""

    name: str
    uid: str


@dataclass(frozen=True)
class DriveSaveResult:
    uid: str
    folder: DriveFolder


@dataclass(frozen=True)
class DriveMediaItem:
    uid: str
    url: str
    name: str
    type: MediaTypeFilter
    timestamp: float
    preview_url: str | None = None


@dataclass(frozen=True)
class DriveFileDetails:
    """A media item with its generation metadata. ``draft_task`` / ``draft_expires_at`` mark a saved Seedance 2.5
    draft: pass ``draft_task`` unchanged as the final's ``draftTask`` param until ``draft_expires_at``."""

    uid: str
    url: str
    name: str
    type: MediaTypeFilter
    timestamp: float
    preview_url: str | None = None
    created_at: Any = None
    model: str | None = None
    prompt: str | None = None
    service: str | None = None
    sub_type: str | None = None
    duration: str | None = None
    user_reaction: UserReaction | None = None
    reference_image_urls: Any = None
    reference_video_url: Any = None
    reference_audio_url: Any = None
    aspect_ratio: Any = None
    resolution: Any = None
    quality: Any = None
    draft_task: dict[str, str] | None = None
    draft_expires_at: str | None = None


@dataclass(frozen=True)
class GenerationFile:
    """A generation's stored attributes; ``ai_sdk_payload`` is every input param (``SdkPayload``)."""

    app_id: str | None = None
    app_type: AppType | None = None
    model: str | None = None
    ai_sdk_payload: dict[str, Any] | None = None
    user_reaction: UserReaction | None = None
    draft_task: dict[str, str] | None = None
    draft_expires_at: str | None = None


@dataclass(frozen=True)
class SaveParams:
    url: str
    name: str
    resource_type: ResourceType
    attributes: dict[str, str] | None = None
    generation: GenerationProvenance | None = None
    """Required to retain full provenance when saving a generated URL separately."""
    preview_url: str | None = None


# ── helpers ──────────────────────────────────────────────────────────
_FLAGS = re.IGNORECASE | re.ASCII
_AUDIO_NAME = re.compile(r"\.(mp3|wav|ogg|aac|flac|m4a)\Z", _FLAGS)
_VIDEO_NAME = re.compile(r"\.(mp4|webm|mov|avi|mkv|m4v|wmv)\Z", _FLAGS)


def _or_empty(value: Any) -> str:
    """``String(value || '')``."""
    return js_string(value) if truthy(value) else ""


def _infer_media_type(file: dict[str, Any]) -> MediaTypeFilter:
    name = _or_empty(js_prop(file, "name"))
    if _AUDIO_NAME.search(name):
        return "audio"
    if _VIDEO_NAME.search(name):
        return "video"
    content_type = js_prop(file, "contentType")
    content_type = js_prop(file, "content") if content_type is None else content_type
    resource_type = _or_empty(content_type.get("resourceType") if isinstance(content_type, dict) else None).upper()
    if resource_type == "VIDEO":
        return "video"
    if resource_type == "AUDIO":
        return "audio"
    return "image"


def _content_resource_types(media_type: MediaTypeFilter | None) -> str:
    return {"image": "PHOTO", "video": "VIDEO", "audio": "AUDIO"}.get(media_type or "", "PHOTO,VIDEO,AUDIO")


def _normalize_url(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    return js_trim(raw) or None


def _parse_attributes(raw: Any) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if not truthy(raw) or not isinstance(raw, (Mapping, list)):
        return attrs
    if isinstance(raw, list):
        for item in raw:  # `{ property, value }` pairs; a missing field stringifies as "undefined"
            js_prop(item, "property")  # null / undefined items throw, as in TS
            attrs[_field_text(item, "property")] = _field_text(item, "value")
    else:
        for key, value in raw.items():
            attrs[key] = js_string(value)
    return attrs


def _field_text(item: Any, key: str) -> str:
    """``String(item[key])``: ``"undefined"`` for a missing key, ``"null"`` for null."""
    if isinstance(item, dict) and key in item:
        return js_string(item[key])
    return "undefined"


def _parse_json_attr(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = js_json_parse(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _parse_draft_info(attrs: Mapping[str, str]) -> dict[str, Any]:
    """The draft fields of a saved Seedance 2.5 draft; empty for any other file or an incomplete reference."""
    draft_task = _parse_json_attr(attrs.get(DRAFT_TASK_ATTR))
    if not is_seedance_draft_task(draft_task):
        return {}
    info: dict[str, Any] = {"draft_task": {"id": draft_task["id"]}}  # type: ignore[index]
    if attrs.get(DRAFT_EXPIRES_AT_ATTR):
        info["draft_expires_at"] = attrs[DRAFT_EXPIRES_AT_ATTR]
    return info


def _parse_reaction(value: str | None) -> UserReaction | None:
    return value if value in ("like", "dislike") else None  # type: ignore[return-value]


def _as_string(value: Any) -> str | None:
    return value if isinstance(value, str) and js_trim(value) else None


def _as_string_array(value: Any) -> list[str] | None:
    return value if isinstance(value, list) and value and all(isinstance(x, str) for x in value) else None


def _extras(attrs: Mapping[str, str]) -> Any:
    """``JSON.parse(attrs.textScript)`` when set (``{}`` when it does not parse)."""
    if not attrs.get("textScript"):
        return {}
    try:
        return js_json_parse(attrs["textScript"])
    except ValueError:
        return {}


# Maps the retired `tool` attribute (legacy app identity) to the current appId / appType.
LEGACY_TOOL_APP: dict[str, tuple[str, AppType]] = {"ai-playground": ("com.picsart.ai-playground", "miniapp")}


def _adapt_legacy_generation(attrs: Mapping[str, str]) -> GenerationFile:
    """Files written before the ``aiSDKPayload`` schema: flat ``model`` / ``prompt`` / ``duration`` attributes plus
    a ``textScript`` JSON blob, folded into ``ai_sdk_payload`` so old files read like new ones."""
    extras = _extras(attrs)
    payload: dict[str, Any] = {"prompt": attrs.get("prompt") or ""}
    for key, source in (("aspectRatio", "aspectRatio"), ("resolution", "resolution")):
        value = _as_string(js_prop(extras, source))
        if value:
            payload[key] = value
    duration = js_prop(extras, "duration")
    duration = attrs.get("duration") if duration is None else duration
    if duration is not None and duration != "":
        payload["duration"] = js_to_number(duration)
    image_urls = _as_string_array(js_prop(extras, "referenceImageUrls"))
    if image_urls:
        payload["imageUrls"] = image_urls
    for key, source in (
        ("videoUrl", "referenceVideoUrl"),
        ("audioUrl", "referenceAudioUrl"),
        ("startFrame", "startFrame"),
        ("endFrame", "endFrame"),
        # The former app-specific fields fold in too (service / subType do not: they derive from the model).
        ("quality", "quality"),
        ("style", "style"),
        ("iterateModel", "iterateModel"),
        ("exploreImageId", "exploreImageId"),
    ):
        value = _as_string(js_prop(extras, source))
        if value:
            payload[key] = value
    app = LEGACY_TOOL_APP.get(attrs["tool"]) if attrs.get("tool") else None
    return GenerationFile(
        app_id=app[0] if app else None,
        app_type=app[1] if app else None,
        model=attrs.get("model") or None,
        ai_sdk_payload=payload,
        user_reaction=_parse_reaction(attrs.get(USER_REACTION_ATTR)),
        **_parse_draft_info(attrs),
    )


def parse_generation(file: Mapping[str, Any]) -> GenerationFile:
    """A Drive file's generation attributes (legacy files go through the adapter)."""
    attrs = _parse_attributes(js_prop(file, "attributes"))
    if not attrs.get("aiSDKPayload"):
        return _adapt_legacy_generation(attrs)
    app_type = attrs.get("appType")
    return GenerationFile(
        app_id=attrs.get("appId") or None,
        app_type=app_type if app_type in ("native", "miniapp") else None,  # type: ignore[arg-type]
        model=attrs.get("model") or None,
        ai_sdk_payload=_parse_json_attr(attrs["aiSDKPayload"]),
        user_reaction=_parse_reaction(attrs.get(USER_REACTION_ATTR)),
        **_parse_draft_info(attrs),
    )


def _attributes_for_save(params: SaveParams) -> dict[str, str]:
    """Generation saves are normalized at the write boundary, not only on reads."""
    attributes = dict(params.attributes or {})
    if params.generation is not None:
        return {**attributes, **build_generation_attributes(params.generation)}
    # Legacy clients supply model / prompt / textScript: persist the canonical form too. Generic URL uploads
    # must not invent generation inputs.
    if _as_string(attributes.get("model")):
        legacy = _adapt_legacy_generation(attributes)
        canonical = _parse_json_attr(attributes.get("aiSDKPayload"))
        payload = (
            canonical
            if canonical is not None and isinstance(canonical.get("prompt"), str)
            else (legacy.ai_sdk_payload or {})
        )
        app = {"appId": legacy.app_id, "appType": legacy.app_type} if legacy.app_id else {}
        return {
            **app,  # type: ignore[dict-item]
            **attributes,
            **build_generation_attributes(GenerationProvenance(model_id=attributes["model"], params=payload)),
        }
    return attributes


def _to_media_item(file: Any) -> DriveMediaItem | None:
    url = _normalize_url(js_prop(file, "sourceUrl"))
    if not url or _or_empty(js_prop(file, "name")).startswith("__"):
        return None
    preview = js_prop(file, "preview")
    updated = js_prop(file, "updatedAt")
    created = js_prop(file, "createdAt")
    uid = js_prop(file, "uid")
    return DriveMediaItem(
        uid=js_string(uid if uid is not None else ""),
        url=url,
        name=_or_empty(js_prop(file, "name")),
        type=_infer_media_type(file),
        preview_url=_normalize_url(preview.get("url") if isinstance(preview, dict) else None),
        timestamp=js_to_number(updated if updated is not None else created if created is not None else 0),
    )


def _to_detailed_item(file: Any) -> DriveFileDetails | None:
    base = _to_media_item(file)
    if base is None:
        return None
    attrs = _parse_attributes(js_prop(file, "attributes"))
    extras = _extras(attrs)
    return DriveFileDetails(
        uid=base.uid,
        url=base.url,
        name=base.name,
        type=base.type,
        timestamp=base.timestamp,
        preview_url=base.preview_url,
        created_at=js_prop(file, "createdAt"),
        model=attrs.get("model"),
        prompt=attrs.get("prompt") or None,
        service=attrs.get("service"),
        sub_type=attrs.get("subType"),
        duration=attrs.get("duration"),
        user_reaction=_parse_reaction(attrs.get(USER_REACTION_ATTR)),
        reference_image_urls=js_prop(extras, "referenceImageUrls"),
        reference_video_url=js_prop(extras, "referenceVideoUrl"),
        reference_audio_url=js_prop(extras, "referenceAudioUrl"),
        aspect_ratio=js_prop(extras, "aspectRatio"),
        resolution=js_prop(extras, "resolution"),
        quality=js_prop(extras, "quality"),
        **_parse_draft_info(attrs),
    )


def _encode_uri_component(text: str) -> str:
    return quote(text, safe="-_.!~*'()")


async def _json(res: httpx.Response) -> Any:
    return js_json_parse(await res.aread())


# ── the client ───────────────────────────────────────────────────────


class DriveClient:
    """``ai.drive``: manages folders, listing, saving and reactions under the root folder."""

    def __init__(self, fetch: AuthenticatedFetch, api_url: str, root_folder_name: str) -> None:
        self._fetch = fetch
        self._api_url = api_url
        self._root_name = root_folder_name
        self._cached_root_uid: Any = None
        self._root_task: asyncio.Future[Any] | None = None

    # ── transport ───────────────────────────────────────────────────────
    async def _post(self, path: str, body: Any) -> httpx.Response:
        init: FetchInit = {
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": js_json_dumps(body),
        }
        return await self._fetch(f"{self._api_url}{path}", init)

    async def _get(self, path: str) -> httpx.Response:
        return await self._fetch(f"{self._api_url}{path}", {})

    # ── folder resolution ───────────────────────────────────────────────
    async def _find_folder_by_path(self, name: str) -> Any:
        try:
            res = await self._get(f"/cloud-storage/v1/me/files-by-path?path={_encode_uri_component(name)}")
            if not res.is_success:
                return None
            data = await _json(res)
            if js_prop(data, "status") != "success":
                return None
            response = js_prop(data, "response")
            file = (response[0] if response else None) if isinstance(response, list) else response
            return file.get("uid") if isinstance(file, dict) else None
        except Exception:
            return None

    @staticmethod
    def _folder_query(parent_uid: Any) -> str:
        return (
            f"parentFolderUid={js_string(parent_uid)}&fileTypes=FOLDER&limit=100"
            if truthy(parent_uid)
            else "fileTypes=FOLDER&limit=100"
        )

    async def _find_folder_in_list(self, name: str, parent_uid: Any = None) -> Any:
        try:
            res = await self._get(f"/cloud-storage/v1/me/files?{self._folder_query(parent_uid)}")
            if not res.is_success:
                return None
            response = js_prop(await _json(res), "response")
            files = response if isinstance(response, list) else []
            wanted = name.lower()
            match = next((f for f in files if _or_empty(js_prop(f, "name")).lower() == wanted), None)
            return match.get("uid") if isinstance(match, dict) else None
        except Exception:
            return None

    async def _create_folder(self, name: str, parent_uid: Any = None) -> Any:
        try:
            body: dict[str, Any] = {"name": name}
            if truthy(parent_uid):
                body["parentFolderUid"] = parent_uid
            res = await self._post("/cloud-storage/v1/me/folders", body)
            if not res.is_success:
                return None
            response = js_prop(await _json(res), "response")
            return response.get("uid") if isinstance(response, dict) else None
        except Exception:
            return None

    async def _resolve_root_folder(self) -> Any:
        by_path = await self._find_folder_by_path(self._root_name)
        if truthy(by_path):
            return by_path
        in_list = await self._find_folder_in_list(self._root_name)
        if truthy(in_list):
            return in_list
        recheck = await self._find_folder_by_path(self._root_name)  # re-check before creating (no duplicates)
        if truthy(recheck):
            return recheck
        return await self._create_folder(self._root_name)

    async def _ensure_root_folder(self) -> Any:
        if truthy(self._cached_root_uid):
            return self._cached_root_uid
        if self._root_task is None:
            task = asyncio.ensure_future(self._resolve_root_folder())
            self._root_task = task
            # Settled in a done callback, never inside the task: under an eager task factory the task can finish
            # before `ensure_future` returns, and the cache must still be cleared after the assignment above.
            task.add_done_callback(self._root_settled)
        shared = self._root_task
        await asyncio.wait({shared})  # one caller's cancellation never cancels the shared resolve
        return shared.result()

    def _root_settled(self, task: asyncio.Future[Any]) -> None:
        if task.cancelled() or task.exception() is not None:
            # Keep the failure for 10 s against retry floods (TS keeps the rejected promise).
            asyncio.get_running_loop().call_later(10, self._forget_root_task, task)
            return
        self._cached_root_uid = task.result()
        self._forget_root_task(task)

    def _forget_root_task(self, task: asyncio.Future[Any]) -> None:
        if self._root_task is task:
            self._root_task = None

    # ── shared internals ────────────────────────────────────────────────
    async def _fetch_folders(self, parent_uid: Any = None) -> list[DriveFolder]:
        try:
            res = await self._get(f"/cloud-storage/v1/me/files?{self._folder_query(parent_uid)}")
            if not res.is_success:
                return []
            response = js_prop(await _json(res), "response")
            files = response if isinstance(response, list) else []
            return [
                DriveFolder(name=js_string(js_prop(f, "name")), uid=js_string(js_prop(f, "uid")))
                for f in files
                if truthy(js_prop(f, "uid")) and truthy(js_prop(f, "name"))
            ]
        except Exception:
            return []

    async def _fetch_media(self, folder_uid: Any, media_type: MediaTypeFilter | None) -> list[Any]:
        try:
            endpoint = "/cloud-storage/v1/me/files" if truthy(folder_uid) else "/cloud-storage/v1/me/flattened-files"
            params = [
                f"parentFolderUid={js_string(folder_uid)}" if truthy(folder_uid) else "",
                "limit=100",
                "sortType=UPDATED",
                "sortOrder=DESC",
                "fileTypes=FILE",
                f"contentResourceTypes={_content_resource_types(media_type)}",
            ]
            res = await self._get(f"{endpoint}?{'&'.join(p for p in params if p)}")
            if not res.is_success:
                return []
            response = js_prop(await _json(res), "response")
            return response if isinstance(response, list) else []
        except Exception:
            return []

    async def _fetch_file_by_uid(self, file_uid: str) -> dict[str, Any] | None:
        try:
            res = await self._get(f"/drive/v1/files/{file_uid}")
            if not res.is_success:
                return None
            file = js_prop(await _json(res), "response")
            return file if isinstance(file, dict) else None
        except Exception:
            return None

    async def _set_reaction(self, file_uid: str, reaction: UserReaction | None) -> bool:
        """Set (or clear, with ``None``) the user's reaction on a file."""
        try:
            init: FetchInit = {
                "method": "PATCH",
                "headers": {"Content-Type": "application/json"},
                "body": js_json_dumps({"attributes": {USER_REACTION_ATTR: reaction}}),
            }
            res = await self._fetch(f"{self._api_url}/drive/v1/files/{file_uid}", init)
            return res.is_success
        except Exception:
            return False

    def _root(self, uid: Any) -> DriveFolder:
        return DriveFolder(name=self._root_name, uid=uid)

    # ── public API ──────────────────────────────────────────────────────
    async def ensure_folder(self, subfolder: str | None = None) -> DriveFolder | None:
        """Ensure ``subfolder`` exists inside the root folder (creating root and subfolder as needed); with no
        argument, just ensure the root folder."""
        root_uid = await self._ensure_root_folder()
        if not truthy(root_uid):
            return None
        if not subfolder:
            return self._root(root_uid)
        existing = await self._find_folder_in_list(subfolder, root_uid)
        if truthy(existing):
            return DriveFolder(name=subfolder, uid=existing)
        created = await self._create_folder(subfolder, root_uid)
        return DriveFolder(name=subfolder, uid=created) if truthy(created) else None

    async def folders(self) -> list[DriveFolder]:
        """The immediate subfolders of the root folder (boards)."""
        root_uid = await self._ensure_root_folder()
        return await self._fetch_folders(root_uid) if truthy(root_uid) else []

    async def _root_and_subfolders(self) -> list[DriveFolder]:
        root_uid = await self._ensure_root_folder()

        async def none() -> list[DriveFolder]:
            return []

        root_level, subfolders = await asyncio.gather(
            self._fetch_folders(), self._fetch_folders(root_uid) if truthy(root_uid) else none()
        )
        return [*root_level, *subfolders]

    async def all_folders(self) -> list[DriveFolder]:
        """Top-level Drive folders plus the root's subfolders, deduplicated by uid."""
        seen: set[str] = set()
        merged: list[DriveFolder] = []
        for folder in await self._root_and_subfolders():
            if folder.uid in seen:
                continue
            seen.add(folder.uid)
            merged.append(folder)
        return merged

    async def find_folder(self, name: str) -> DriveFolder | None:
        """A folder by name (case-insensitive) across the top level and the root's subfolders."""
        if name.lower() == self._root_name.lower():
            uid = await self._ensure_root_folder()
            return self._root(uid) if truthy(uid) else None
        wanted = name.lower()
        return next((f for f in await self._root_and_subfolders() if f.name.lower() == wanted), None)

    async def list(
        self,
        *,
        folder: DriveFolder | None = None,
        type: MediaTypeFilter | None = None,  # noqa: A002
    ) -> list[DriveMediaItem]:
        """Media items in ``folder`` (all folders, flattened, without one), newest first, optionally by type."""
        files = await self._fetch_media(folder.uid if folder is not None else None, type)
        return [item for item in map(_to_media_item, files) if item is not None]

    async def list_detailed(
        self,
        *,
        folder: DriveFolder | None = None,
        type: MediaTypeFilter | None = None,  # noqa: A002
    ) -> builtins.list[DriveFileDetails]:  # `list` is the method above here
        """Like :meth:`list`, with each file's generation metadata."""
        files = await self._fetch_media(folder.uid if folder is not None else None, type)
        return [item for item in map(_to_detailed_item, files) if item is not None]

    async def get_generation(self, file_uid: str) -> GenerationFile | None:
        """A single generation's stored attributes."""
        file = await self._fetch_file_by_uid(file_uid)
        return parse_generation(file) if file is not None else None

    async def save(self, params: SaveParams, folder: DriveFolder | None = None) -> DriveSaveResult | None:
        """Save an asset to ``folder`` (the root folder by default); ``None`` on failure."""
        target_uid = folder.uid if folder is not None else await self._ensure_root_folder()
        if not truthy(target_uid):
            return None
        target = folder if folder is not None else self._root(target_uid)
        body: dict[str, Any] = {
            "name": params.name,
            "sourceUrl": params.url,
            "parentFolderUid": target_uid,
            "content": {"type": "STANDALONE", "resourceType": params.resource_type, "sourcePlatform": "WEB"},
            "preview": {"url": params.preview_url or params.url, "width": 1024, "height": 1024},
            "attributes": [{"property": k, "value": v} for k, v in _attributes_for_save(params).items()],
        }
        try:
            res = await self._post("/cloud-storage/v1/me/files", body)
            if res.status_code == 400:
                # Retry under a safe name when the original trips the restricted-keywords filter.
                if "restricted_keywords" not in (await res.aread()).decode("utf-8", errors="replace"):
                    return None
                ext = params.name.split(".")[-1] or "png"
                body["name"] = f"ai-generation-{int(time.time() * 1000)}.{ext}"
                res = await self._post("/cloud-storage/v1/me/files", body)
            if not res.is_success:
                return None
            file = js_prop(await _json(res), "response")
            uid = file.get("uid") if isinstance(file, dict) else None
            return DriveSaveResult(uid=uid, folder=target) if truthy(uid) else None  # type: ignore[arg-type]
        except Exception:
            return None

    def build_save_params(
        self,
        url: str,
        model_id: str,
        model_name: str,
        mode: str,
        prompt_or_params: str | Mapping[str, Any] | None = None,
    ) -> SaveParams:
        """Standard save params for a generation result (a prompt, or the full params)."""
        params = {"prompt": prompt_or_params} if isinstance(prompt_or_params, str) else dict(prompt_or_params or {})
        prompt = js_string(params["prompt"]) if params.get("prompt") is not None else ""
        return SaveParams(
            url=url,
            name=build_filename(prompt, mode, {"url": url}),
            resource_type=infer_resource_type(mode),
            attributes={
                "tool": "ai-sdk",
                **build_generation_attributes(GenerationProvenance(model_id=model_id, params=params)),
                # Cut like the payload's copy: the full prompt would push this attribute past Drive's limit.
                "prompt": cut_stored_prompt(prompt),
                "service": model_name,
            },
        )

    async def add_reaction(self, file_uid: str, reaction: UserReaction) -> bool:
        """Set a like / dislike reaction on a file."""
        return await self._set_reaction(file_uid, reaction)

    async def remove_reaction(self, file_uid: str) -> bool:
        """Clear the reaction on a file."""
        return await self._set_reaction(file_uid, None)


def create_drive_client(fetch: AuthenticatedFetch, api_url: str, root_folder_name: str) -> DriveClient:
    return DriveClient(fetch, api_url, root_folder_name)


__all__ = [
    "DriveClient",
    "DriveFileDetails",
    "DriveFolder",
    "DriveMediaItem",
    "DriveSaveResult",
    "GenerationFile",
    "MediaTypeFilter",
    "ResourceType",
    "SaveParams",
    "UserReaction",
    "create_drive_client",
    "parse_generation",
]
