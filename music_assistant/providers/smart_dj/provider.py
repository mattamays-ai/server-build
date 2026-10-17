"""Smart DJ queue intelligence backed by Music Assistant audio analysis and Musicae."""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web
from music_assistant_models.auth import Scope
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import (
    ConfigEntryType,
    ExternalID,
    MediaType,
    PlaybackState,
    QueueOption,
)

from music_assistant.controllers.player_queues.helpers import build_queue_item
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.controllers.streams.constants import BufferMode, BufferSize
from music_assistant.helpers.app_vars import app_var
from music_assistant.helpers.audio import decoded_pcm_format
from music_assistant.models.plugin import PluginProvider

from .engine import (
    MODES,
    DJControls,
    DJMode,
    SignalControl,
    _merge_track,
    beam_optimize,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant

CONF_RAPIDAPI_KEY = "rapidapi_key"
MUSICAE_HOST = "dj-track-audio-analysis-api.p.rapidapi.com"
MUSICAE_BASE = f"https://{MUSICAE_HOST}"
CACHE_TTL = 86400.0

# Signals whose weight may be tuned per call; clap joins them once the
# audio-analysis pipeline carries the embedding (models/audio_analysis.py).
CLAP_SIGNAL = "clap"

# Route the injected UI script is served from, and the tag spliced into index.html.
UI_ROUTE = "/smart-dj-ui.js"
UI_SCRIPT_TAG = '<script src="/smart-dj-ui.js" defer></script>'


AUTOMIX_MAX_CROSSFADE = 60.0

# Silent preroll analyzer: cap per track (the analysis providers themselves cap
# at 10 minutes) and a poll interval for detecting stored results.
PREROLL_ANALYSIS_TIMEOUT = 600
PREROLL_POLL_SECONDS = 5


def _automix_durations(tracks: list[dict[str, Any]], control: DJControls) -> dict[str, float]:
    """
    Derive per-track crossfade lengths (seconds) from the plan's transition bars.

    The transition INTO a track lasts as long as that track's bars (per-track
    override first, then the global setting) at the track's own tempo. Tracks
    without usable bpm get no pin and keep the engine's default length.
    """
    durations: dict[str, float] = {}
    for track in tracks:
        analysis = track.get("analysis") or {}
        raw_bpm = analysis.get("bpm") or track.get("bpm")
        if raw_bpm is None:
            continue
        try:
            bpm = float(raw_bpm)
        except TypeError, ValueError:
            continue
        if bpm <= 0:
            continue
        bars = (
            control.transition_bars_overrides.get(str(track.get("queue_item_id")))
            or control.transition_bars
        )
        try:
            beats_per_bar = int(analysis.get("beats_per_bar") or 4)
        except TypeError, ValueError:
            beats_per_bar = 4
        if beats_per_bar <= 0:
            beats_per_bar = 4
        # one bar lasts beats_per_bar beats at `bpm` beats a minute; capped so a
        # slow 32-bar pin can't demand an absurd PCM hold-back
        durations[str(track.get("queue_item_id"))] = round(
            min(int(bars) * beats_per_bar * 60.0 / bpm, AUTOMIX_MAX_CROSSFADE), 2
        )
    return durations


def _automix_blend_prefs(
    tracks: list[dict[str, Any]], control: DJControls
) -> dict[str, tuple[str, str]]:
    """
    Pin (transition_style, beat_strictness) for every AutoMix-pinned track.

    Same keys as the duration pins (the blend INTO each track reads its own
    pref). Style "auto" + strictness "normal" means engine defaults, which the
    streams-side getter collapses to no pin at all.
    """
    return {
        str(track.get("queue_item_id")): (control.transition_style, control.beat_strictness)
        for track in tracks
        if track.get("queue_item_id") is not None
    }


def _camelot_from_key(key: str | None, mode: str | None) -> str | None:
    """Convert Music Assistant key/mode to Camelot notation."""
    if not key or not mode:
        return None
    minor = mode.lower() in {"minor", "min", "m"}
    minor_keys = {
        "Ab": "1A",
        "Eb": "2A",
        "Bb": "3A",
        "F": "4A",
        "C": "5A",
        "G": "6A",
        "D": "7A",
        "A": "8A",
        "E": "9A",
        "B": "10A",
        "F#": "11A",
        "C#": "12A",
    }
    major_keys = {
        "B": "1B",
        "F#": "2B",
        "C#": "3B",
        "Ab": "4B",
        "Eb": "5B",
        "Bb": "6B",
        "F": "7B",
        "C": "8B",
        "G": "9B",
        "D": "10B",
        "A": "11B",
        "E": "12B",
    }
    normalized = key.replace("\u266d", "b").replace("\u266f", "#")
    return (minor_keys if minor else major_keys).get(normalized)


def _snapshot_start_index(
    snapshot: list[dict[str, Any]],
    raw_queue_item_ids: list[str],
    current_index: int,
) -> int | None:
    """Find the first analyzable queue item at or after the current index."""
    index_by_id = {
        str(item["queue_item_id"]): index
        for index, item in enumerate(snapshot)
        if item.get("queue_item_id") is not None
    }
    return next(
        (
            index_by_id[str(queue_item_id)]
            for queue_item_id in raw_queue_item_ids[max(0, current_index) :]
            if str(queue_item_id) in index_by_id
        ),
        None,
    )


class SmartDJError(RuntimeError):
    """Named, UI-facing Smart DJ error; ``code`` is a stable identifier."""

    def __init__(self, code: str, detail: str = "") -> None:
        """Store the stable ``code`` and build the UI-facing message."""
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class MusicaeRateLimited(SmartDJError):
    """Musicae rejected the request with 429 after bounded retries."""


# Bounded Musicae client: explicit timeout, limited in-flight requests, and a
# small retry budget for 429/5xx/network so one quota wall cannot fan out into
# 100 simultaneous failures (external audit H-4).
MUSICAE_CONCURRENCY = 4
MUSICAE_TIMEOUT_S = 20.0
MUSICAE_MAX_ATTEMPTS = 3  # 1 try + 2 retries, never more

# A wholesale queue write-back is destructive by nature; refuse (loudly,
# BEFORE any mutation) to run it against an absurdly large queue.
APPLY_MAX_QUEUE_ITEMS = 5000

# mix_playlist mixes at most this many playlist tracks (the planner window);
# the response reports the truncation instead of hiding it (audit M-3).
MIX_PLAYLIST_MAX_TRACKS = 100


def _first_present(*vals: Any) -> Any:
    """First value that is not None. Unlike ``or`` this keeps a legit 0/""."""
    for val in vals:
        if val is not None:
            return val
    return None


def _finite_in(
    value: Any,
    lo: float | None = None,
    hi: float | None = None,
    *,
    positive: bool = False,
) -> float | None:
    """
    Return float(value) when finite and in bounds, else None.

    Vendor payloads are untrusted: NaN/Inf/out-of-range/non-numeric values
    must be dropped (to None), never passed into the planner.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    if positive and as_float <= 0:
        return None
    if lo is not None and as_float < lo:
        return None
    if hi is not None and as_float > hi:
        return None
    return as_float


def _int_control(
    raw: dict[str, Any],
    name: str,
    default: int,
    allowed: set[int] | None = None,
    lo: int | None = None,
    hi: int | None = None,
) -> int:
    """Coerce a control value to int, raising a NAMED error when malformed."""
    value_raw = raw.get(name, default)
    try:
        value = int(value_raw)
    except TypeError, ValueError:
        raise SmartDJError("invalid_control", f"{name}={value_raw!r}") from None
    if allowed is not None and value not in allowed:
        raise SmartDJError("invalid_control", f"{name}={value}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise SmartDJError("invalid_control", f"{name}={value}")
    return value


def _float_control(
    raw: dict[str, Any],
    name: str,
    default: Any = None,
    lo: float | None = None,
    hi: float | None = None,
) -> float | None:
    """Coerce an optional control value to float, raising a NAMED error."""
    value_raw = raw.get(name, default)
    if value_raw is None:
        return None
    try:
        value = float(value_raw)
    except TypeError, ValueError:
        raise SmartDJError("invalid_control", f"{name}={value_raw!r}") from None
    if (
        not math.isfinite(value)
        or (lo is not None and value < lo)
        or (hi is not None and value > hi)
    ):
        raise SmartDJError("invalid_control", f"{name}={value}")
    return value


def _signal_weight(value: Any) -> float:
    try:
        weight = float(value)
    except TypeError, ValueError:
        raise SmartDJError("invalid_control", f"weight={value!r}") from None
    if not math.isfinite(weight) or weight < 0.0:
        raise SmartDJError("invalid_control", f"weight={value!r}")
    return max(0.0, weight)


def _bpm_tolerance(value: Any) -> float:
    try:
        tol = float(value)
    except TypeError, ValueError:
        raise SmartDJError("invalid_control", f"bpm_tolerance={value!r}") from None
    if not math.isfinite(tol):
        raise SmartDJError("invalid_control", f"bpm_tolerance={value!r}")
    return max(0.01, min(1.0, tol))


class SmartDJProvider(PluginProvider):
    """Native Smart DJ controller and Musicae enrichment client."""

    async def analyze(
        self, queue_id: str, limit: int = 40, analysis_provider: str = "auto"
    ) -> dict[str, Any]:
        """Analyze the active queue starting at its current playable item."""
        queue = self.mass.player_queues.get(queue_id)
        if queue is None:
            raise RuntimeError(f"Queue not found: {queue_id}")

        # _queue_snapshot omits queue items without resolvable media provider IDs.
        # Resolve the current start against the unfiltered queue first, then find
        # that queue-item ID in the compact snapshot so skipped entries do not
        # shift the current-index calculation.
        raw_items = self._read_full_queue(queue_id)
        current_index = max(0, int(queue.current_index or 0))
        snapshot = await self._queue_snapshot(queue_id)
        start = _snapshot_start_index(
            snapshot,
            [str(item.queue_item_id) for item in raw_items],
            current_index,
        )
        if start is None:
            items: list[dict[str, Any]] = []
        else:
            items = snapshot[start : start + max(1, min(limit, 100))]

        tasks = [
            asyncio.create_task(
                self._analysis(
                    item["item_id"],
                    item["provider"],
                    {key: item[key] for key in ("genre", "genres", "explicit") if key in item},
                    analysis_provider,
                )
            )
            for item in items
        ]
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        analyzed: list[dict[str, Any]] = []
        for item, result in zip(items, analyses, strict=True):
            analysis = None if isinstance(result, Exception) else result
            if isinstance(result, Exception):
                self.logger.debug("Smart DJ analysis failed for %s: %s", item["item_id"], result)
            analyzed.append({**item, "analysis": analysis})

        current = analyzed[0]["analysis"] if analyzed and analyzed[0].get("analysis") else None
        analysis_failed = sum(1 for res in analyses if isinstance(res, Exception))
        return {
            "queue_id": queue_id,
            "tracks": analyzed,
            "current": current,
            "queue_item_count": len(raw_items),
            "analysis_failed": analysis_failed,
            "analysis_rate_limited": sum(
                1 for res in analyses if isinstance(res, MusicaeRateLimited)
            ),
            "degraded": analysis_failed > 0,
        }

    async def preroll_queue_analysis(self, queue_id: str, limit: int = 10) -> dict[str, Any]:
        """
        Silently analyze upcoming queue tracks that have no analysis yet.

        Feeds each unanalyzed upcoming track's audio through the analysis
        providers WITHOUT playback (no player, no audible output): the track's
        PCM stream is fetched, pumped through a private buffer, and the stored
        result is picked up by the DJ engine when the track reaches the blend.
        Runs strictly sequentially with one source slot at a time, so it stays
        out of the way of active playback.
        """
        queue = self.mass.player_queues.get(queue_id)
        if queue is None:
            raise RuntimeError(f"Queue not found: {queue_id}")
        if not self.mass.streams.audio_analysis.providers:
            raise RuntimeError("No audio analysis providers are loaded")
        items = self._read_full_queue(queue_id)
        current_index = max(0, int(queue.current_index or 0))
        # skip the playing track and its preloaded next: playback analyzes those
        upcoming = [
            item for item in items[current_index + 1 :] if item.media_type == MediaType.TRACK
        ]
        prerolled: list[str] = []
        failed: dict[str, str] = {}
        for item in upcoming:
            if len(prerolled) >= max(1, min(int(limit), 100)):
                break
            try:
                done = await self._preroll_track_analysis(item)
            except Exception as err:
                self.logger.debug("Smart DJ preroll failed for %s: %s", item.name, err)
                failed[str(item.queue_item_id)] = str(err)
                continue
            if done:
                prerolled.append(str(item.queue_item_id))
        return {"queue_id": queue_id, "prerolled": prerolled, "failed": failed}

    async def complete_mix(
        self,
        queue_id: str,
        count: int = 5,
        controls: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Complete the queue: append compatible tracks sourced from Musicae.

        Seeds Musicae's radio endpoint (a continuous mix chained from one seed
        track) with the queue's tail track, resolves the returned Spotify ids
        to Music Assistant tracks, and appends them at the end of the queue.
        When the queue has AutoMix pins, the appended tracks are analyzed and
        pinned too, so the blend into them is handled like the rest of the mix.
        """
        queue = self.mass.player_queues.get(queue_id)
        if queue is None:
            raise RuntimeError(f"Queue not found: {queue_id}")
        snapshot = await self._queue_snapshot(queue_id)
        if not snapshot:
            raise RuntimeError("Queue has no resolvable tracks to seed from")
        seed = snapshot[-1]
        spotify_instances = self.mass.get_provider_instances("spotify")
        if not spotify_instances:
            raise RuntimeError("Complete mix requires a Spotify provider instance")
        spotify_instance = spotify_instances[0].instance_id

        # Musicae radio: primary path per the 2026 docs, seeded by one track id.
        # The endpoint shape is path-parameterized (/v2/tracks/{id}/radio); if
        # that 404s, try the flat variant once. A wrong path raises with both
        # response bodies so the shape is visible in the error.
        limit = max(1, min(int(count), 50))
        errors = []
        data = None
        for path, params in (
            (f"/v2/tracks/{seed['item_id']}/radio", {"limit": limit}),
            ("/v2/radio", {"seed_id": seed["item_id"], "limit": limit}),
        ):
            try:
                data = await self._request(path, params)
                break
            except RuntimeError as err:
                errors.append(f"{path}: {err}")
        if not data:
            raise RuntimeError("Musicae radio failed - " + " | ".join(errors))
        raw_tracks = data.get("tracks") if isinstance(data, dict) else None
        if not isinstance(raw_tracks, list):
            raise RuntimeError(  # noqa: TRY004 - vendor shape, not a type error
                f"Unexpected Musicae radio response shape: {str(data)[:300]}"
            )

        tracks: list[Any] = []
        seen: set[str] = set()
        for raw in raw_tracks:
            track_id = raw.get("id") if isinstance(raw, dict) else None
            if not track_id or track_id == seed["item_id"] or track_id in seen:
                continue
            seen.add(track_id)
            try:
                track = await self.mass.music.tracks.get(track_id, spotify_instance)
            except Exception as err:
                self.logger.debug("Musicae radio track %s not resolvable: %s", track_id, err)
                continue
            tracks.append(track)
            if len(tracks) >= limit:
                break
        if not tracks:
            raise RuntimeError("Musicae radio returned no tracks resolvable via Spotify")

        queue_items = [build_queue_item(queue_id, track) for track in tracks]
        insert_at = len(self._read_full_queue(queue_id))
        await self.mass.player_queues.load(
            queue_id,
            queue_items,
            insert_at_index=insert_at,
            keep_remaining=True,
            keep_played=True,
        )
        # fire-and-forget the silent preroll so the appended tracks arrive
        # analyzed before they reach a blend; AWAITING it here would block this
        # command for minutes (each unanalyzed track is a full PCM fetch).
        # automix_pinned reports whether the queue is running AutoMix, so the
        # caller knows the appended tracks will get the crossfade path (their
        # per-item lengths come from the queue's pins at blend time).
        automix_pinned = 0
        if getattr(queue, "crossfade_enabled", False):
            automix_pinned = len(queue_items)
            self.mass.create_task(self.preroll_queue_analysis(queue_id, len(queue_items)))
        return {
            "queue_id": queue_id,
            "seed": seed.get("name"),
            "added": [track.name for track in tracks],
            "automix_pinned": automix_pinned,
        }

    async def analyze_library(
        self,
        limit: int = 500,
        source: str = "recently_added",
    ) -> dict[str, Any]:
        """
        Batch-enrich library tracks with Musicae analysis, most recent first.

        Pulls up to ``limit`` library tracks (newest-added first, or
        recently-played), resolves each to a Musicae id (Spotify id when the
        track maps to Spotify, else its ISRC), and fetches the full analysis
        payload in batches of 5 (the documented per-request maximum), storing
        everything in the provider cache so ranking can use it without
        per-track latency. Stops early on a 429 so quota is never burned
        against a hard rate limit.
        """
        if source not in {"recently_added", "recently_played", "all"}:
            raise RuntimeError("source must be one of recently_added, recently_played, all")
        order = {
            "recently_added": "timestamp_added_desc",
            "recently_played": "last_played_desc",
            "all": "sort_name",
        }[source]
        cap = max(1, min(int(limit), 500))
        tracks = await self.mass.music.tracks.library_items(
            favorite=None,
            search=None,
            limit=cap,
            offset=0,
            order_by=order,
            summary=False,
        )
        # resolve each track to (cache_key, musicae_id); prefer a Spotify id,
        # fall back to ISRC (Musicae resolves by Spotify ID or ISRC)
        targets: list[tuple[str, str]] = []
        now = time.time()
        for track in tracks:
            key = None
            if track.provider:
                candidate = f"{track.provider}:{track.item_id}"
                cached = self._cache.get(candidate)
                if not (cached and now - cached[0] < CACHE_TTL):
                    key = (candidate, track.item_id)
            if key is None:
                isrc = track.get_external_id(ExternalID.ISRC)
                if isrc:
                    cache_key = f"isrc:{isrc}"
                    cached = self._cache.get(cache_key)
                    if not (cached and now - cached[0] < CACHE_TTL):
                        key = (cache_key, isrc)
            if key is not None:
                targets.append(key)

        batch_size = 5  # documented /v2/audio-analysis batch maximum
        fetched = 0
        quota_exhausted = False
        errors: list[str] = []
        for start in range(0, len(targets), batch_size):
            chunk = targets[start : start + batch_size]
            try:
                data = await self._request(
                    "/v2/audio-analysis/" + ",".join(musicae_id for _, musicae_id in chunk),
                    {},
                )
            except RuntimeError as err:
                if "429" in str(err):
                    quota_exhausted = True
                    break
                errors.append(str(err)[:200])
                continue
            raw_entries: Any = data
            entries = (
                raw_entries if isinstance(raw_entries, list) else (raw_entries.get("data") or [])
            )
            if not isinstance(entries, list):
                entries = [entries]
            for (cache_key, _), entry in zip(chunk, entries, strict=False):
                if not isinstance(entry, dict) or not entry:
                    continue
                self._cache[cache_key] = (now, self._normalize_musicae(entry))
                fetched += 1
            await asyncio.sleep(0.5)  # stay friendly to the plan's rate limit
        await self._save_cache()
        return {
            "source": source,
            "scanned": len(tracks),
            "resolved": len(targets),
            "fetched": fetched,
            "quota_exhausted": quota_exhausted,
            "errors": errors[:5],
        }

    async def mix_playlist(  # noqa: PLR0915 - linear pipeline by design
        self,
        queue_id: str,
        playlist_provider: str,
        playlist_item_id: str,
        bpm_tolerance: float = 0.08,
        prefer_keys: bool = True,
        preserve_variety: bool = True,
        mode: str = "ai_dj",
        controls: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Play a playlist on the given queue in Smart DJ (beat/key) order."""
        raw = dict(controls or {})
        analysis_provider = str(raw.get("analysis_provider", "auto"))
        automix_info: dict[str, Any] | None = None
        control = self._parse_controls(raw, bpm_tolerance, prefer_keys, preserve_variety, mode)

        playlist_tracks: list[Any] = [
            item
            async for item in self.mass.music.playlists.tracks(playlist_item_id, playlist_provider)
        ]
        playlist_total = len(playlist_tracks)
        truncated = playlist_total > MIX_PLAYLIST_MAX_TRACKS
        if truncated:
            playlist_tracks = playlist_tracks[:MIX_PLAYLIST_MAX_TRACKS]
            self.logger.warning(
                "Smart DJ: playlist %s has %d tracks; mixing the first %d",
                playlist_item_id,
                playlist_total,
                MIX_PLAYLIST_MAX_TRACKS,
            )
        if len(playlist_tracks) < 2:
            raise RuntimeError("Playlist needs at least 2 tracks to mix")

        # Resolve lightweight item mappings to full tracks so genre metadata
        # (needed for analysis) is available.
        resolved: list[Any] = []
        for track in playlist_tracks:
            if getattr(track, "metadata", None) is None and getattr(track, "item_id", None):
                with suppress(Exception):
                    track = await self.mass.music.tracks.get(  # noqa: PLW2901
                        track.item_id, track.provider
                    )
            if getattr(track, "item_id", None) and (
                getattr(track, "provider", None) or getattr(track, "provider_instance", None)
            ):
                resolved.append(track)
        if len(resolved) < 2:
            raise RuntimeError("Playlist tracks could not be resolved for mixing")

        analysis_rows: list[dict[str, Any]] = []
        media_by_id: dict[str, Any] = {}
        for idx, track in enumerate(resolved):
            metadata = getattr(track, "metadata", None)
            genres = list(getattr(metadata, "genres", None) or [])
            mix_id = f"mix-{idx}"
            analysis_rows.append(
                {
                    "queue_item_id": mix_id,
                    "name": track.name,
                    "artist": getattr(track, "artist_str", None) or getattr(track, "artist", None),
                    "item_id": track.item_id,
                    "provider": getattr(track, "provider", None)
                    or getattr(track, "provider_instance", None),
                    "genre": genres[0] if genres else None,
                    "genres": genres,
                    "explicit": getattr(metadata, "explicit", None) if metadata else None,
                }
            )
            media_by_id[mix_id] = track

        tasks = [
            asyncio.create_task(
                self._analysis(
                    item["item_id"],
                    item["provider"],
                    {k: item[k] for k in ("genre", "genres", "explicit") if k in item},
                    analysis_provider,
                )
            )
            for item in analysis_rows
        ]
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        analyzed: list[dict[str, Any]] = []
        for item, result in zip(analysis_rows, analyses, strict=True):
            analysis = None if isinstance(result, Exception) else result
            if isinstance(result, Exception):
                self.logger.debug("Smart DJ analysis failed for %s: %s", item["item_id"], result)
            analyzed.append({**item, "analysis": analysis})

        preset = MODES.get(mode, MODES["ai_dj"])
        selected_mode = DJMode(
            preset.name,
            _bpm_tolerance(bpm_tolerance),
            preset.energy_direction,
            preset.variety,
            preset.weights,
        )
        queue = self.mass.player_queues.get(queue_id)
        if queue is None:
            raise RuntimeError(f"Queue not found: {queue_id}")

        playing = queue.state in (PlaybackState.PLAYING, PlaybackState.PAUSED)
        items = self._read_full_queue(queue_id)
        current_index = max(0, int(queue.current_index or 0))
        current_media = (
            items[current_index].media_item if playing and current_index < len(items) else None
        )
        if len(items) > APPLY_MAX_QUEUE_ITEMS:
            raise SmartDJError(
                "queue_too_large",
                f"{len(items)} queue items exceeds the rebuild ceiling of "
                f"{APPLY_MAX_QUEUE_ITEMS}; refusing a destructive rebuild",
            )
        current_media_id = str(getattr(current_media, "item_id", "") or "")
        playlist_entry_ids = {str(entry["item_id"]) for entry in analyzed}

        if playing and current_media_id in playlist_entry_ids:
            # RESUME REBUILD: the queue is playing THIS playlist mid-stream.
            # Anchor the plan at the track that is playing right now (it keeps
            # playing, untouched) and reorder only the not-yet-played tail --
            # instead of replacing the queue and restarting from the top.
            anchor = next(entry for entry in analyzed if str(entry["item_id"]) == current_media_id)
            played_ids = {
                str(getattr(item.media_item, "item_id", "") or "")
                for item in items[: current_index + 1]
            }
            remaining = [entry for entry in analyzed if str(entry["item_id"]) not in played_ids]
            optimized = beam_optimize(
                remaining,
                _merge_track(anchor),
                selected_mode,
                controls=control,
                beam_width=max(4, min(96, control.lookahead * 3)),
            )
            if not optimized:
                raise RuntimeError(
                    "Hard requirements are impossible with the remaining playlist tracks"
                )
            # Map planned entries to queue items: reuse the existing upcoming
            # queue item when the track is already enqueued (keeps streamdetails
            # and pinned state alive), build a fresh one when it is not.
            by_media_id: dict[str, Any] = {}
            for item in items[current_index + 1 :]:
                media_id = str(getattr(item.media_item, "item_id", "") or "")
                by_media_id.setdefault(media_id, item)
            ranked_items: list[Any] = []
            id_map: dict[str, str] = {}  # plan's synthetic mix-N -> real queue_item_id
            for entry in optimized:
                qitem = by_media_id.pop(str(entry["item_id"]), None)
                if qitem is None:
                    qitem = build_queue_item(queue_id, media_by_id[entry["queue_item_id"]])
                id_map[str(entry["queue_item_id"])] = str(qitem.queue_item_id)
                ranked_items.append(qitem)
            assert len({item.queue_item_id for item in ranked_items}) == len(ranked_items), (
                "Queue integrity check failed: duplicate queue item in plan"
            )
            ranked_ids = {item.queue_item_id for item in ranked_items}
            remainder = [
                item for item in items[current_index + 1 :] if item.queue_item_id not in ranked_ids
            ]
            self.mass.player_queues.update_items(
                queue_id, items[: current_index + 1] + ranked_items + remainder
            )
            if control.automix_enabled:
                self.mass.player_queues.set_crossfade(queue_id, True)
                # pins are read back by the blend engine keyed on the REAL
                # queue_item_id of the incoming track, so remap the plan's
                # synthetic ids before pinning
                self.mass.streams.audio.set_automix_durations(
                    {
                        id_map[mix_id]: secs
                        for mix_id, secs in _automix_durations(optimized, control).items()
                        if mix_id in id_map
                    }
                )
                self.mass.streams.audio.set_automix_blend_prefs(
                    {
                        id_map[mix_id]: pref
                        for mix_id, pref in _automix_blend_prefs(optimized, control).items()
                        if mix_id in id_map
                    }
                )
                self.mass.create_task(self.preroll_queue_analysis(queue_id, 10))
                automix_info = {
                    "enabled": True,
                    "mode": "smart" if self.mass.streams.smart_fades_available else "standard",
                }
            return {
                "queue_id": queue_id,
                "playlist_provider": playlist_provider,
                "playlist_item_id": playlist_item_id,
                "resumed": True,
                "playlist_total": playlist_total,
                "truncated": truncated,
                "current_item_id": current_media_id,
                "count": len(optimized),
                "tracks": optimized,
                "automix": automix_info,
            }

        # FRESH MIX: nothing playing (or the playing track is not from this
        # playlist) -- replace the queue with the fully optimized playlist.
        optimized = beam_optimize(
            analyzed,
            None,
            selected_mode,
            controls=control,
            beam_width=max(4, min(96, control.lookahead * 3)),
        )
        if not optimized:
            raise RuntimeError("Hard requirements are impossible with the current playlist")
        ordered_media = [media_by_id[track["queue_item_id"]] for track in optimized]
        self.logger.info(
            "Smart DJ: mixing playlist %s/%s (%d tracks) on queue %s",
            playlist_provider,
            playlist_item_id,
            len(ordered_media),
            queue_id,
        )
        await self.mass.player_queues.play_media(queue_id, ordered_media, option=QueueOption.PLAY)
        if control.automix_enabled:
            # Real AutoMix execution on the freshly mixed queue (smart fades
            # when analysis is available, standard crossfade otherwise).
            self.mass.player_queues.set_crossfade(queue_id, True)
            # pins must key the REAL queue_item_ids the blend engine reads back;
            # re-read the freshly built queue and map media item ids to them
            fresh_by_media: dict[str, Any] = {}
            for item in self._read_full_queue(queue_id):
                media_id = str(getattr(item.media_item, "item_id", "") or "")
                fresh_by_media.setdefault(media_id, item)
            fresh_ids = {
                str(entry["queue_item_id"]): str(
                    fresh_by_media[str(entry["item_id"])].queue_item_id
                )
                for entry in analyzed
                if str(entry["item_id"]) in fresh_by_media
            }
            self.mass.streams.audio.set_automix_durations(
                {
                    fresh_ids[mix_id]: secs
                    for mix_id, secs in _automix_durations(analyzed, control).items()
                    if mix_id in fresh_ids
                }
            )
            self.mass.streams.audio.set_automix_blend_prefs(
                {
                    fresh_ids[mix_id]: pref
                    for mix_id, pref in _automix_blend_prefs(analyzed, control).items()
                    if mix_id in fresh_ids
                }
            )
            # silent background preroll: analyze upcoming tracks so blends
            # improve as the queue plays (no audible output)
            self.mass.create_task(self.preroll_queue_analysis(queue_id, 10))
            automix_info = {
                "enabled": True,
                "mode": "smart" if self.mass.streams.smart_fades_available else "standard",
            }
        return {
            "queue_id": queue_id,
            "playlist_provider": playlist_provider,
            "playlist_item_id": playlist_item_id,
            "resumed": False,
            "playlist_total": playlist_total,
            "truncated": truncated,
            "count": len(ordered_media),
            "tracks": optimized,
            "automix": automix_info,
        }

    async def rank_queue(  # noqa: PLR0915 - linear pipeline by design
        self,
        queue_id: str,
        bpm_tolerance: float = 0.08,
        prefer_keys: bool = True,
        preserve_variety: bool = True,
        mode: str = "ai_dj",
        controls: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Optimize the upcoming queue under an explicit user control contract."""
        raw = dict(controls or {})
        apply = bool(raw.pop("apply", False))
        analysis_provider = str(raw.get("analysis_provider", "auto"))
        snapshot = await self.analyze(queue_id, analysis_provider=analysis_provider)
        tracks = snapshot["tracks"]
        if len(tracks) < 2:
            return snapshot

        control = self._parse_controls(raw, bpm_tolerance, prefer_keys, preserve_variety, mode)
        if not control.smart_reorder_enabled:
            if apply and control.automix_enabled:
                # Real AutoMix execution: the queue-level crossfade toggle is a
                # supported runtime command, so flipping it is safe even when no
                # reorder is requested.
                self.mass.player_queues.set_crossfade(queue_id, True)
                self.mass.streams.audio.set_automix_durations(
                    _automix_durations(tracks[1:], control)
                )
                self.mass.streams.audio.set_automix_blend_prefs(
                    _automix_blend_prefs(tracks[1:], control)
                )
                self.mass.create_task(self.preroll_queue_analysis(queue_id, 10))
                automix_pin = {
                    "enabled": True,
                    "mode": "smart" if self.mass.streams.smart_fades_available else "standard",
                }
            return {
                "queue_id": queue_id,
                "tracks": tracks[1:],
                # "applied" means THE QUEUE WAS MUTATED. Reorder-disabled
                # never touches the order; at most it pinned automix
                # durations/prefs, which is reported separately below.
                "applied": False,
                "queue_changed": False,
                "automix_applied": automix_pin is not None,
                "reason": "smart_reorder_disabled",
                "automix": automix_pin,
                "settings": {"controls": raw, "smart_reorder_enabled": False},
            }

        selected_mode = MODES.get(mode, MODES["ai_dj"])
        selected_mode = DJMode(
            selected_mode.name,
            _bpm_tolerance(bpm_tolerance),
            selected_mode.energy_direction,
            selected_mode.variety,
            selected_mode.weights,
        )
        optimized = beam_optimize(
            tracks[1:],
            _merge_track(tracks[0]) if tracks else snapshot["current"],
            selected_mode,
            controls=control,
            beam_width=max(4, min(96, control.lookahead * 3)),
        )
        if not optimized and tracks[1:]:
            raise RuntimeError("Hard requirements are impossible with the current queue")

        automix_info: dict[str, Any] | None = None
        if apply:
            queue = self.mass.player_queues.get(queue_id)
            if queue is None:
                raise RuntimeError(f"Queue not found: {queue_id}")
            items = self._read_full_queue(queue_id)
            by_id = {item.queue_item_id: item for item in items}
            if len(items) > APPLY_MAX_QUEUE_ITEMS:
                raise SmartDJError(
                    "queue_too_large",
                    f"{len(items)} queue items exceeds the apply ceiling of "
                    f"{APPLY_MAX_QUEUE_ITEMS}; refusing a destructive reorder",
                )
            prefix_len = (queue.current_index or 0) + 1
            prefix = items[:prefix_len]
            # The plan was built from a snapshot taken before analysis/planning;
            # the player may have advanced since. The fresh queue is ground
            # truth: drop planned tracks that already slid into the now-playing
            # prefix (or vanished) instead of failing the apply. Only an
            # internally inconsistent partition is a real error.
            prefix_ids = {item.queue_item_id for item in prefix}
            ranked_ids = [
                track["queue_item_id"]
                for track in optimized
                if track["queue_item_id"] not in prefix_ids and track["queue_item_id"] in by_id
            ]
            dropped = len(optimized) - len(ranked_ids)
            if dropped:
                self.logger.info(
                    "Smart DJ apply: dropped %d planned track(s) that are no "
                    "longer upcoming on queue %s",
                    dropped,
                    queue_id,
                )
            ranked = [by_id[item_id] for item_id in ranked_ids]
            if len({item.queue_item_id for item in ranked}) != len(ranked):
                raise RuntimeError("Queue integrity check failed")
            ranked_set = set(ranked_ids)
            remainder = [
                item for item in items[prefix_len:] if item.queue_item_id not in ranked_set
            ]
            if len(prefix) + len(ranked) + len(remainder) != len(items):
                raise RuntimeError("Queue integrity check failed")
            self.mass.player_queues.update_items(queue_id, prefix + ranked + remainder)
            if not control.automix_enabled:
                # automix off in this plan: release any previously pinned lengths
                self.mass.streams.audio.set_automix_durations({})
                self.mass.streams.audio.set_automix_blend_prefs({})
            if control.automix_enabled:
                # Real AutoMix execution: hand the queue to the streams engine's
                # crossfade path (smart fades when analysis is available, standard
                # otherwise). A plan with automix disabled never touches the toggle.
                self.mass.player_queues.set_crossfade(queue_id, True)
                self.mass.streams.audio.set_automix_durations(
                    _automix_durations(optimized, control)
                )
                self.mass.streams.audio.set_automix_blend_prefs(
                    _automix_blend_prefs(optimized, control)
                )
                self.mass.create_task(self.preroll_queue_analysis(queue_id, 10))
                automix_info = {
                    "enabled": True,
                    "mode": "smart" if self.mass.streams.smart_fades_available else "standard",
                }
        return {
            "queue_id": queue_id,
            "current_item_id": tracks[0]["queue_item_id"],
            "tracks": optimized,
            "optimized_window": len(tracks),
            "queue_item_count": snapshot.get("queue_item_count"),
            # "applied" means THE QUEUE WAS MUTATED and nothing else
            "applied": bool(apply),
            "queue_changed": bool(apply),
            "automix_applied": automix_info is not None,
            "automix": automix_info,
            "settings": {
                "mode": mode,
                "bpm_tolerance": bpm_tolerance,
                "preserve_variety": preserve_variety,
                "controls": raw,
            },
        }

    async def capabilities(self) -> dict[str, Any]:
        """Describe available Smart DJ analysis and mixing capabilities."""
        analysis_controller = getattr(self.mass.streams, "audio_analysis", None)
        return {
            "analysis": {
                "music_assistant": analysis_controller is not None,
                "musicae": bool(self._key()),
            },
            "mixing": {
                "smart_fades": True,
                "transition_planner": True,
                # the single-deck engine has NO vocal protection, bass-EQ
                # management, or tempo planning; advertising True made the FE
                # print "Reported" for features that do not exist (audit M-4)
                "vocal_protection": False,
                "bass_eq_management": False,
                "tempo_planning": False,
            },
            "analysis_providers": {
                "music_assistant": analysis_controller is not None,
                "musicae": bool(self._key()),
                "fallback_policy": ["music_assistant", "musicae"],
            },
            "user_control": {
                "states": ["hard", "soft", "disabled"],
                "transition_bars": [4, 8, 16, 32],
                "lookahead": {"min": 1, "max": 96},
                # beam width = lookahead x 3, capped at 96: the top slider stop
                # already buys a 96-wide search (the FE label says so now)
                "lookahead_beam_cap": 96,
                "transition_styles": ["auto", "full_blend", "energy", "quick_cut"],
                "beat_strictness": ["strict", "normal", "aggressive"],
                "complete_mix": True,
                "preroll_analysis": True,
                "signals": [
                    "bpm",
                    "key",
                    "energy",
                    "danceability",
                    "loudness",
                    "genre",
                    "artist_spacing",
                    "momentum",
                    CLAP_SIGNAL,
                ],
            },
        }

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Expose the Musicae RapidAPI key so it can be set in the provider UI."""
        return (
            ConfigEntry(
                key=CONF_RAPIDAPI_KEY,
                type=ConfigEntryType.SECURE_STRING,
                default_value=app_var(CONF_RAPIDAPI_KEY),
                required=False,
            ),
        )

    async def status(self) -> dict[str, Any]:
        """Return provider readiness without revealing credentials."""
        return {
            "configured": bool(self._key()),
            "musicae_host": MUSICAE_HOST,
            "cached_tracks": len(self._cache),
        }

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[Any],
    ) -> None:
        """Set up the provider and its analysis cache and API handles."""
        super().__init__(mass, manifest, config, supported_features)
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._session: aiohttp.ClientSession | None = None
        self._handles: list[Any] = []
        self._musicae_sem = asyncio.Semaphore(MUSICAE_CONCURRENCY)
        self._save_lock = asyncio.Lock()
        self._cache_file = Path(self.mass.storage_path) / "smart_dj" / "analysis_cache.json"
        # UI injection state: the patched index.html and the restored original.
        self._tmp_index_path: str | None = None
        self._orig_index_path: str | None = None
        self._unregister_ui_route: Any = None

    async def loaded_in_mass(self) -> None:
        """Register the Smart DJ API commands and load the persistent cache."""
        await self._load_cache()
        handlers = (
            # rank_queue can replace the whole upcoming queue, so it is member-only:
            # LIBRARY_WRITE is the member/guest boundary (user/admin/service hold it,
            # guests don't), preserving the old required_role="user" exclusion of
            # guest accounts. analyze is ALSO member-only: it can spend Musicae
            # quota and writes the on-disk cache, so a guest must never trigger
            # it (external audit H-2). Only genuinely read-only commands
            # (status, capabilities) stay at required_scope=None.
            ("smart_dj/analyze", self.analyze, Scope.LIBRARY_WRITE),
            ("smart_dj/rank_queue", self.rank_queue, Scope.LIBRARY_WRITE),
            ("smart_dj/mix_playlist", self.mix_playlist, Scope.LIBRARY_WRITE),
            ("smart_dj/status", self.status, None),
            ("smart_dj/capabilities", self.capabilities, None),
            (
                "smart_dj/preroll_queue_analysis",
                self.preroll_queue_analysis,
                Scope.LIBRARY_WRITE,
            ),
            ("smart_dj/complete_mix", self.complete_mix, Scope.LIBRARY_WRITE),
            ("smart_dj/analyze_library", self.analyze_library, Scope.LIBRARY_WRITE),
        )
        for command, handler, required_scope in handlers:
            self._handles.append(
                self.mass.register_api_command(command, handler, required_scope=required_scope)
            )
        self._setup_ui()

    async def unload(self, is_removed: bool = False) -> None:
        """Close the HTTP client, drop the UI, and unregister commands."""
        self._teardown_ui()
        for handle in self._handles:
            handle()
        self._handles.clear()
        if self._session:
            await self._session.close()
            self._session = None
        await super().unload(is_removed)

    def _read_full_queue(self, queue_id: str, page: int = 500) -> list[Any]:
        """
        Read every queue item via paged reads.

        A single limit-capped read silently truncates long queues, and the
        wholesale write-back of an apply would then DELETE the unread tail
        (external audit C-1). Paged reads make the operation total.
        """
        out: list[Any] = []
        offset = 0
        while True:
            batch = list(self.mass.player_queues.items(queue_id, limit=page, offset=offset))
            out.extend(batch)
            if len(batch) < page:
                return out
            offset += page

    async def _preroll_track_analysis(self, queue_item: Any) -> bool:
        """Silently analyze one queue item. Returns True when analysis was stored."""
        media = queue_item.media_item
        media_provider = getattr(media, "provider", None) or getattr(
            media, "provider_instance", None
        )
        media_item_id = getattr(media, "item_id", None)
        if not media_provider or not media_item_id:
            return False
        existing = await self.mass.streams.audio_analysis.get_audio_analysis(
            item_id=media_item_id,
            provider_instance_id_or_domain=media_provider,
        )
        if existing is not None:
            return False
        streamdetails = await self.mass.streams.audio.get_stream_details(queue_item)
        if streamdetails.is_realtime:
            return False
        pcm_format = decoded_pcm_format(streamdetails)
        buffer = AudioBuffer(pcm_format, BufferSize.BALANCED, BufferMode.SEEKABLE)
        buffer.fill(
            self.mass.streams.audio.get_media_stream(streamdetails, pcm_format),
            source_name="smart_dj_preroll",
        )
        try:
            await self.mass.streams.audio_analysis.start_analysis(buffer, streamdetails)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + PREROLL_ANALYSIS_TIMEOUT
            while loop.time() < deadline and not buffer.cancelled and not buffer.eof:
                await asyncio.sleep(PREROLL_POLL_SECONDS)
                stored = await self.mass.streams.audio_analysis.get_audio_analysis(
                    item_id=media_item_id,
                    provider_instance_id_or_domain=media_provider,
                )
                if stored is not None:
                    return True
            return False
        finally:
            # tears down the producer and (via the buffer's cancel callback)
            # evicts any still-active analysis session for this item
            await buffer.clear()

    def _normalize_musicae(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Normalize one Musicae analysis entry into the cache shape."""
        instrumental_raw = entry.get("instrumental")
        instrumentalness_raw = entry.get("instrumentalness")
        instrumental_flag = (
            instrumental_raw
            if isinstance(instrumental_raw, bool)
            else (
                instrumentalness_raw >= 0.5
                if isinstance(instrumentalness_raw, (int, float))
                else None
            )
        )
        return {
            # first-present (NOT `or`): a legitimate 0 must not fall through
            # to a fallback field; numerics are finite/range validated
            "bpm": _finite_in(
                _first_present(entry.get("bpm"), (entry.get("rhythm") or {}).get("bpm")),
                positive=True,
            ),
            "key": _first_present(
                entry.get("key"),
                (entry.get("harmony") or {}).get("key"),
                entry.get("musical_key"),
            ),
            "mode": _first_present(entry.get("mode"), (entry.get("harmony") or {}).get("mode")),
            "camelot": _first_present(
                entry.get("camelot"), (entry.get("harmony") or {}).get("camelot")
            ),
            "energy": _finite_in(
                _first_present(entry.get("energy"), (entry.get("score") or {}).get("energy")),
                0.0,
                1.0,
            ),
            "danceability": _finite_in(entry.get("danceability"), 0.0, 1.0),
            "valence": _finite_in(entry.get("valence"), 0.0, 1.0),
            "arousal": _finite_in(entry.get("arousal"), 0.0, 1.0),
            "loudness": _finite_in(
                _first_present(entry.get("loudness"), entry.get("loudness_integrated"))
            ),
            "beats_per_bar": _finite_in(
                _first_present(
                    entry.get("beats_per_bar"),
                    (entry.get("rhythm") or {}).get("time_signature"),
                ),
                1.0,
                12.0,
            ),
            "instrumental": instrumental_flag,
            "instrumentalness": instrumentalness_raw,
            "source": "musicae",
            "raw": entry,
        }

    def _parse_controls(
        self,
        raw: dict[str, Any],
        bpm_tolerance: float,
        prefer_keys: bool,
        preserve_variety: bool,
        mode: str,
    ) -> DJControls:
        """Build DJControls from raw API controls plus ranking defaults."""

        def signal(name: str, fallback: str = "soft") -> SignalControl:
            value = raw.get(name, fallback)
            if isinstance(value, dict):
                return SignalControl(
                    state=str(value.get("state", fallback)),
                    weight=_signal_weight(value.get("weight", 1.0)),
                )
            return SignalControl(state=str(value), weight=1.0)

        transition_bars = _int_control(raw, "transition_bars", 8, allowed={4, 8, 16, 32})
        overrides_raw = raw.get("transition_bars_overrides") or {}
        if not isinstance(overrides_raw, dict):
            raise RuntimeError(  # noqa: TRY004 - control contract, not a type error
                "transition_bars_overrides must be an object of track id -> bars"
            )
        overrides: dict[str, int] = {}
        for track_id, bars in overrides_raw.items():
            try:
                bars_int = int(bars)
            except TypeError, ValueError:
                raise RuntimeError(f"Invalid transition bars value for {track_id}: {bars!r}")
            if bars_int not in {0, 4, 8, 16, 32}:
                raise RuntimeError(
                    f"Invalid transition bars value for {track_id}: {bars_int} "
                    "(must be 0 (hard cut), 4, 8, 16, or 32)"
                )
            overrides[str(track_id)] = bars_int
        key_relation = str(raw.get("key_relation", "compatible"))
        if key_relation not in {"any", "same", "compatible", "soft"}:
            raise RuntimeError("key_relation must be one of any, same, compatible, soft")
        transition_style = str(raw.get("transition_style", "auto"))
        if transition_style not in {"auto", "full_blend", "energy", "quick_cut"}:
            raise RuntimeError(
                "transition_style must be one of auto, full_blend, energy, quick_cut"
            )
        beat_strictness = str(raw.get("beat_strictness", "normal"))
        if beat_strictness not in {"strict", "normal", "aggressive"}:
            raise RuntimeError("beat_strictness must be one of strict, normal, aggressive")
        aggr = _float_control(raw, "transition_aggressiveness", 0.5, 0.0, 1.0)
        return DJControls(
            bpm=signal("bpm"),
            key=signal("key", "soft" if prefer_keys else "disabled"),
            energy=signal("energy"),
            danceability=signal("danceability"),
            loudness=signal("loudness"),
            genre=signal("genre"),
            artist_spacing=signal("artist_spacing", "soft" if preserve_variety else "disabled"),
            momentum=signal("momentum"),
            clap=signal(CLAP_SIGNAL),
            bpm_min=_float_control(raw, "bpm_min", lo=0.0),
            bpm_max=_float_control(raw, "bpm_max", lo=0.0),
            max_bpm_jump=_float_control(raw, "max_bpm_jump", lo=0.0),
            key_relation=key_relation,
            max_artist_repeat=_int_control(raw, "max_artist_repeat", 1, lo=0, hi=10),
            instrumental=str(raw.get("instrumental", "any")),
            explicit=str(raw.get("explicit", "allow")),
            transition_bars=transition_bars,
            transition_bars_overrides=overrides,
            automix_enabled=bool(raw.get("automix_enabled", False)),
            smart_reorder_enabled=bool(raw.get("smart_reorder_enabled", True)),
            transition_style=transition_style,
            beat_strictness=beat_strictness,
            # the FE slider spans 1-96 (beam width = lookahead x 3, capped
            # at 96), so the backend accepts the full range it advertises
            lookahead=_int_control(raw, "lookahead", 8, lo=1, hi=96),
            transition_aggressiveness=0.5 if aggr is None else aggr,
            required_ids=frozenset(str(x) for x in raw.get("required_ids", [])),
            excluded_ids=frozenset(str(x) for x in raw.get("excluded_ids", [])),
            fixed_ids=frozenset(str(x) for x in raw.get("fixed_ids", [])),
            end_track_id=str(raw["end_track_id"]) if raw.get("end_track_id") else None,
        )

    def _setup_ui(self) -> None:
        """Serve the UI script and inject its tag into the served index.html."""
        ui_js_path = Path(__file__).with_name("ui.js")
        if not ui_js_path.exists():
            return
        # Read the script once; serve it from memory on every request.
        try:
            js_content = ui_js_path.read_text(encoding="utf-8")
        except OSError:
            return

        async def _serve_js(_request: web.Request) -> web.Response:
            return web.Response(
                text=js_content,
                content_type="application/javascript",
                headers={"Cache-Control": "no-cache"},
            )

        with suppress(RuntimeError):  # already registered on provider reload
            self._unregister_ui_route = self.mass.webserver.register_dynamic_route(
                UI_ROUTE, _serve_js, "GET"
            )

        # Inject the script tag by serving a patched copy of index.html.
        orig = getattr(self.mass.webserver, "_index_path", None)
        if not orig or not Path(orig).exists():
            # a core refactor can remove/rename the private hook; the
            # failure mode is a silently missing menu, so make it LOUD
            self.logger.warning(
                "Smart DJ UI injection skipped: webserver._index_path is "
                "missing or unreadable; the injected Smart DJ menu will be "
                "unavailable"
            )
            return
        self._orig_index_path = orig
        try:
            html = Path(orig).read_text(encoding="utf-8")
        except OSError:
            return
        if UI_ROUTE in html:
            return  # already injected
        # crash leftovers from a previous run would otherwise accumulate
        with suppress(OSError):
            for stale in Path(tempfile.gettempdir()).glob("ma_smart_dj_*.html"):
                stale.unlink(missing_ok=True)
        html = html.replace("</body>", f"{UI_SCRIPT_TAG}</body>")
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".html",
                delete=False,
                prefix="ma_smart_dj_",
            ) as tmp:
                tmp.write(html)
                self._tmp_index_path = tmp.name
            self.mass.webserver._index_path = self._tmp_index_path
        except OSError as err:
            self.logger.warning("Could not create patched index.html: %s", err)

    def _teardown_ui(self) -> None:
        """Restore the original index.html and drop the dynamic route."""
        if self._orig_index_path:
            self.mass.webserver._index_path = self._orig_index_path
            self._orig_index_path = None
        if self._tmp_index_path:
            with suppress(OSError):
                Path(self._tmp_index_path).unlink(missing_ok=True)
            self._tmp_index_path = None
        if callable(self._unregister_ui_route):
            self._unregister_ui_route()
            self._unregister_ui_route = None

    async def _load_cache(self) -> None:
        """Load the persistent analysis cache."""
        try:
            if self._cache_file.exists():
                raw = await asyncio.to_thread(self._cache_file.read_text)
                data = json.loads(raw)
                now = asyncio.get_running_loop().time()
                if isinstance(data, dict):
                    loaded: dict[str, tuple[float, dict[str, Any]]] = {}
                    for key, entry in data.items():
                        if isinstance(entry, dict) and isinstance(entry.get("value"), dict):
                            age = float(entry.get("saved_at", 0.0))
                            timestamp = now - max(0.0, time.time() - age) if age else now
                            if now - timestamp < CACHE_TTL:
                                loaded[key] = (timestamp, entry["value"])
                        elif isinstance(entry, dict):
                            loaded[key] = (now, entry)
                    self._cache = loaded
        except (OSError, ValueError, TypeError) as err:
            self.logger.warning("Could not load Smart DJ cache: %s", err)

    async def _save_cache(self) -> None:
        """
        Persist analysis cache atomically.

        A lock serializes saves and each save uses a UNIQUE temp file: the
        old fixed ``analysis_cache.tmp`` name meant two overlapping saves
        could collide on the same path (external audit M-1).
        """
        try:
            await asyncio.to_thread(self._cache_file.parent.mkdir, parents=True, exist_ok=True)
            wall_now = time.time()
            loop_now = asyncio.get_running_loop().time()
            payload = {
                key: {
                    "saved_at": wall_now - max(0.0, loop_now - timestamp),
                    "value": value,
                }
                for key, (timestamp, value) in self._cache.items()
            }
            async with self._save_lock:
                await asyncio.to_thread(self._atomic_write_cache, payload)
        except OSError as err:
            self.logger.warning("Could not save Smart DJ cache: %s", err)

    def _atomic_write_cache(self, payload: dict[str, Any]) -> None:
        """Write payload to a unique temp file, fsync, then rename over."""
        fd, tmp_name = tempfile.mkstemp(dir=self._cache_file.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            Path(tmp_name).replace(self._cache_file)
        except BaseException:
            with suppress(OSError):
                Path(tmp_name).unlink()
            raise

    def _key(self) -> str | None:
        value = self.config.get_value(CONF_RAPIDAPI_KEY)
        # call-time env fallback (same pattern as Spotify's app_var reads): the
        # add-on wrapper injects MASS_APP_VAR_RAPIDAPI_KEY, and a config default
        # only prefills the UI — never rely on the parse path alone.
        return str(value) if value else (app_var(CONF_RAPIDAPI_KEY) or None)

    async def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Call Musicae without exposing credentials to clients.

        Bounded concurrency, explicit timeout, and bounded retry for
        429/5xx/network errors only. 401/403 never retry. Retries honor a
        small numeric Retry-After (capped) so a quota wall cannot stall a
        sweep for minutes (external audit H-4).
        """
        key = self._key()
        if not key:
            raise RuntimeError("Smart DJ requires a Musicae RapidAPI key")
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=MUSICAE_TIMEOUT_S)
            )
        headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": MUSICAE_HOST}
        url = f"{MUSICAE_BASE}{path}"
        last_err = "unknown"
        for attempt in range(MUSICAE_MAX_ATTEMPTS):
            retry_after: float | None = None
            try:
                async with (
                    self._musicae_sem,
                    self._session.get(url, params=params, headers=headers) as response,
                ):
                    if response.status in (401, 403):
                        body = await response.text()
                        raise SmartDJError(
                            "musicae_auth",
                            f"Musicae request failed ({response.status}): {body[:300]}",
                        )
                    if response.status == 429 or response.status >= 500:
                        body = await response.text()
                        last_err = f"({response.status}): {body[:300]}"
                        ra = response.headers.get("Retry-After", "")
                        if ra.replace(".", "", 1).isdigit():
                            retry_after = float(ra)
                    elif response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            f"Musicae request failed ({response.status}): {body[:300]}"
                        )
                    else:
                        data = await response.json()
                        return data if isinstance(data, dict) else {}
            except RuntimeError, SmartDJError:
                raise
            except (aiohttp.ClientError, TimeoutError) as err:
                last_err = f"network: {err}"
            if attempt < MUSICAE_MAX_ATTEMPTS - 1:
                delay = min(
                    5.0,
                    retry_after if retry_after is not None else float(attempt + 1),
                )
                self.logger.warning(
                    "Musicae %s - retry %d/%d in %.1fs",
                    last_err,
                    attempt + 1,
                    MUSICAE_MAX_ATTEMPTS - 1,
                    delay,
                )
                await asyncio.sleep(delay)
        if last_err.startswith("(429"):
            raise MusicaeRateLimited(last_err)
        raise RuntimeError(
            f"Musicae request failed after {MUSICAE_MAX_ATTEMPTS} attempts: {last_err}"
        )

    async def _analysis(
        self,
        item_id: str,
        provider: str,
        metadata: dict[str, Any] | None,
        analysis_provider: str = "auto",
    ) -> dict[str, Any] | None:
        """Return normalized analysis for one track, preferring Music Assistant."""
        if analysis_provider in {"auto", "music_assistant"}:
            try:
                analysis = await self.mass.streams.audio_analysis.get_audio_analysis(
                    item_id=item_id,
                    provider_instance_id_or_domain=provider,
                )
                if analysis is not None:
                    return {
                        "bpm": analysis.bpm,
                        "key": analysis.key,
                        "mode": analysis.mode,
                        "energy": analysis.energy,
                        "danceability": analysis.danceability,
                        "valence": analysis.valence,
                        "arousal": analysis.arousal,
                        "loudness": analysis.loudness_integrated,
                        "beats_per_bar": analysis.beats_per_bar,
                        "beats": analysis.beats,
                        "downbeats": analysis.downbeats,
                        "rms_energy": analysis.rms_energy,
                        "spectral_centroid": analysis.spectral_centroid,
                        "instrumental": (
                            None
                            if analysis.instrumentalness is None
                            else analysis.instrumentalness >= 0.5
                        ),
                        "instrumentalness": analysis.instrumentalness,
                        "camelot": _camelot_from_key(analysis.key, analysis.mode),
                        "clap_embedding": analysis.clap_embedding,
                        "source": "music_assistant",
                        "sources": dict.fromkeys(
                            (
                                "bpm",
                                "key",
                                "camelot",
                                "energy",
                                "danceability",
                                "loudness",
                                "beats_per_bar",
                                "beats",
                                "downbeats",
                                "instrumental",
                                "clap_embedding",
                            ),
                            "music_assistant",
                        ),
                        **(metadata or {}),
                    }
            except Exception as err:
                self.logger.debug(
                    "MA audio analysis unavailable for %s/%s: %s", provider, item_id, err
                )

        if analysis_provider == "music_assistant":
            return None

        provider_obj = self.mass.get_provider(provider)
        provider_domain = getattr(provider_obj, "domain", provider)
        if provider_domain != "spotify":
            return None
        now = asyncio.get_running_loop().time()
        cache_key = f"{provider}:{item_id}"
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < CACHE_TTL:
            return cached[1]

        data = await self._request(f"/v2/audio-analysis/{item_id}", {})
        result = data.get("data") if isinstance(data.get("data"), dict) else data
        if not isinstance(result, dict):
            return None
        instrumental_raw = result.get("instrumental")
        instrumentalness_raw = result.get("instrumentalness")
        instrumental_flag = (
            instrumental_raw
            if isinstance(instrumental_raw, bool)
            else (
                instrumentalness_raw >= 0.5
                if isinstance(instrumentalness_raw, (int, float))
                else None
            )
        )
        normalized = {
            "bpm": _finite_in(result.get("bpm"), positive=True),
            "key": _first_present(result.get("key"), result.get("musical_key")),
            "mode": result.get("mode"),
            "camelot": _first_present(
                result.get("camelot"),
                _camelot_from_key(
                    _first_present(result.get("key"), result.get("musical_key")),
                    result.get("mode"),
                ),
            ),
            "energy": _finite_in(result.get("energy"), 0.0, 1.0),
            "danceability": _finite_in(result.get("danceability"), 0.0, 1.0),
            "valence": _finite_in(result.get("valence"), 0.0, 1.0),
            "arousal": _finite_in(result.get("arousal"), 0.0, 1.0),
            "loudness": _finite_in(
                _first_present(result.get("loudness"), result.get("loudness_integrated"))
            ),
            "beats_per_bar": _finite_in(
                _first_present(result.get("beats_per_bar"), result.get("time_signature")),
                1.0,
                12.0,
            ),
            "beats": result.get("beats"),
            "downbeats": result.get("downbeats"),
            # first-present keeps a legitimate 0.0/[] from falling through
            "rms_energy": _first_present(result.get("rms_energy"), result.get("waveform")),
            "spectral_centroid": result.get("spectral_centroid"),
            "instrumental": instrumental_flag,
            "instrumentalness": result.get("instrumentalness"),
            "clap_embedding": result.get("clap_embedding"),
            "source": "musicae",
            "sources": dict.fromkeys(
                (
                    "bpm",
                    "key",
                    "camelot",
                    "energy",
                    "danceability",
                    "loudness",
                    "beats_per_bar",
                    "beats",
                    "downbeats",
                    "instrumental",
                    "clap_embedding",
                ),
                "musicae",
            ),
            "raw": result,
            **(metadata or {}),
        }
        self._cache[cache_key] = (now, normalized)
        await self._save_cache()
        return normalized

    async def _queue_snapshot(self, queue_id: str) -> list[dict[str, Any]]:
        """Return a compact queue snapshot."""
        items = self._read_full_queue(queue_id)
        result: list[dict[str, Any]] = []
        for item in items:
            media = item.media_item
            provider = getattr(media, "provider", None) or getattr(media, "provider_instance", None)
            item_id = getattr(media, "item_id", None)
            if not provider or not item_id:
                continue
            metadata_obj = getattr(media, "metadata", None)
            genres = getattr(metadata_obj, "genres", None) if metadata_obj else None
            genre_list = list(genres or [])
            result.append(
                {
                    "queue_item_id": item.queue_item_id,
                    "name": item.name,
                    "artist": getattr(media, "artist_str", None) or getattr(media, "artist", None),
                    "item_id": item_id,
                    "provider": provider,
                    "genre": genre_list[0] if genre_list else None,
                    "genres": list(genres or []),
                    "explicit": getattr(metadata_obj, "explicit", None) if metadata_obj else None,
                }
            )
        return result
