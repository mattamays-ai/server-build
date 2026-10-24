"""Smart DJ queue intelligence backed by Music Assistant audio analysis and Musicae."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import tempfile
import time
from collections import deque
from contextlib import suppress
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web
from music_assistant_models.auth import Scope
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import (
    ConfigEntryType,
    EventType,
    ExternalID,
    MediaType,
    PlaybackState,
    QueueOption,
)

from music_assistant.constants import (
    CONF_BACKGROUND_SCAN_CONCURRENCY,
    DEFAULT_BACKGROUND_SCAN_CONCURRENCY,
)
from music_assistant.controllers.player_queues.helpers import build_queue_item
from music_assistant.controllers.streams.audio_analysis import (
    SONIC_ANALYSIS_DOMAIN,
    has_real_analysis,
)
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.controllers.streams.constants import BufferMode, BufferSize
from music_assistant.controllers.streams.smart_fades import stems as stems_ducking
from music_assistant.helpers.app_vars import app_var
from music_assistant.helpers.audio import decoded_pcm_format
from music_assistant.models.music_provider import MusicProvider
from music_assistant.models.plugin import PluginProvider

from . import quick_index
from .engine import (
    ARC_CLIMB_MAX,
    ARC_RESET_DROP,
    MODES,
    DJControls,
    DJMode,
    HardRequirementsImpossible,
    SignalControl,
    _merge_track,
    beam_optimize,
    camelot_affinity,
)
from .quick_index import (
    ENERGY_BANDS,
    INDEX_FILENAME,
    INDEX_VERSION,
    VOCAL_DENSE_DENSITY,
    VOCAL_TIER_NAMES,
    QuickIndex,
    analysis_from_index_row,
    compatible_buckets,
    derive_row,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.event import MassEvent
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant

CONF_RAPIDAPI_KEY = "rapidapi_key"
# Pre-analysis on arrival (0.0.28 lane B): toggle for analyzing newly
# added library tracks in the background (writer in get_config_entries,
# reader in _preanalysis_enabled - shipped as a pair).
CONF_PREANALYSIS_ENABLED = "preanalysis_enabled"
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

# Library-wide local analysis sweep safeguards. The sweep must never make
# the box feel bad: it defers to active playback, runs at most a handful of
# tracks at a time, and aborts rather than hammering a broken provider.
LIBRARY_SCAN_MAX_TRACKS = 10000
LIBRARY_SCAN_POLL_SECONDS = 10.0
LIBRARY_SCAN_MAX_FAILURES = 50
LIBRARY_SCAN_MAX_ERRORS_SHOWN = 5

# Hard ceiling for ONE sweep attempt (streamdetails fetch + buffer fill +
# analysis wait). The analyzer's own PREROLL_ANALYSIS_TIMEOUT is 600s - far too
# slow for a sweep: a hung producer must not stall a track slot for 10 minutes.
# Cancellation is safe: _analyze_media_silent's finally always runs
# buffer.clear(), which cancels the producer and (via the buffer's registered
# cancel callback) evicts the still-active analysis session.
ANALYSIS_ATTEMPT_TIMEOUT = 45

# Pre-analysis on arrival (0.0.28 lane B): the pending queue is BOUNDED - when
# full, the oldest pending track is dropped (logged) so a burst of additions
# can never grow memory without limit. Bulk provider syncs do not emit
# per-item events (SUPPRESS_MEDIA_ITEM_UPDATES), so the queue only ever sees
# genuinely new, individually added tracks.
PREANALYSIS_QUEUE_MAX = 50


def _sweep_track_name(track: Any) -> str:
    """Best-effort display name for a sweep failure row."""
    return str(getattr(track, "name", None) or getattr(track, "item_id", None) or "unknown track")


def _record_sweep_failure(state: dict[str, Any], item_id: str, name: str, reason: str) -> None:
    """Record one sweep failure honestly: capped errors list + last-10 rows."""
    if len(state["errors"]) < LIBRARY_SCAN_MAX_ERRORS_SHOWN:
        state["errors"].append(f"{reason}: {item_id} ({name})"[:200])
    rows: list[dict[str, str]] = state["recent_failures"]
    rows.append({"item_id": str(item_id), "name": str(name), "reason": str(reason)[:200]})
    del rows[:-10]


def _automix_durations(
    tracks: list[dict[str, Any]],
    control: DJControls,
    previous: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Derive per-track crossfade lengths (seconds) from the plan's transition bars.

    The transition INTO a track lasts as long as that track's bars: per-track
    override first, then the DERIVED per-transition count (0.0.28 lane C,
    with ``previous`` as the first track's outgoing predecessor), then the
    global setting - at the track's own tempo. Tracks without usable bpm get
    no pin and keep the engine's default length.
    """
    durations: dict[str, float] = {}
    outgoing = previous
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
        override = control.transition_bars_overrides.get(str(track.get("queue_item_id")))
        if override is not None:
            # the historical ``or``-fallback semantics for a 0 (hard cut)
            # override are preserved: a 0.0s pin is not a supported blend
            bars = override or control.transition_bars
        else:
            bars = derive_transition_bars(outgoing, track, control)[0]
        outgoing = track
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


# genre keyword -> (transition_style, beat_strictness) applied when the user's
# style pin is "auto"; the first keyword hit wins, an explicit user pick always
# wins, and an unmatched genre keeps the engine defaults
GENRE_PROFILES: tuple[tuple[str, tuple[str, str]], ...] = (
    # beatmatchable, four-on-the-floor material: the full DJ blend is the point
    ("house", ("full_blend", "normal")),
    ("techno", ("full_blend", "normal")),
    ("trance", ("full_blend", "normal")),
    ("electro", ("full_blend", "normal")),
    ("disco", ("full_blend", "normal")),
    ("dance", ("full_blend", "normal")),
    ("funk", ("full_blend", "aggressive")),
    # halved/double-time meters: blendable but only within a tight stretch range
    ("drum and bass", ("full_blend", "strict")),
    ("drum & bass", ("full_blend", "strict")),
    ("jungle", ("full_blend", "strict")),
    ("dubstep", ("full_blend", "strict")),
    # vocal-dense or live material: nothing to beatmatch, hand over sequentially
    ("hip hop", ("radio", "strict")),
    ("hip-hop", ("radio", "strict")),
    ("rap", ("radio", "strict")),
    ("rock", ("radio", "aggressive")),
    ("pop", ("radio", "aggressive")),
    ("punk", ("radio", "aggressive")),
    ("metal", ("radio", "aggressive")),
    ("indie", ("radio", "aggressive")),
    ("jazz", ("radio", "aggressive")),
    ("classical", ("radio", "aggressive")),
    ("ambient", ("radio", "aggressive")),
    ("soundtrack", ("radio", "aggressive")),
    ("folk", ("radio", "aggressive")),
    ("soul", ("radio", "aggressive")),
)


def _genre_profile(genre: str | None) -> tuple[str, str] | None:
    """Match a track's genre string against the per-genre profile table."""
    if not genre:
        return None
    lowered = genre.lower()
    for keyword, profile in GENRE_PROFILES:
        if keyword in lowered:
            return profile
    return None


# Stable snake_case rule ids emitted by the pair-style resolver. The FE keys
# plain-language definitions to these ids (server half of the 0.0.27
# definitions contract) -- they are API surface, never rename them.
RULE_QUICK_CUT_BLOCKED = "quick_cut_blocked"
RULE_DOUBLE_DROP_NON_DNB = "double_drop_non_dnb"
RULE_BACKSPIN_NEVER_AUTO = "backspin_never_auto"
RULE_TRANSITION_BARS_CAPPED = "transition_bars_capped"

# Per-transition stems veto reasons (0.0.28 lane D). Stable snake_case ids the
# FE reads from the mix response's per-transition metadata
# ("stems_veto_reason"); they are API surface, never rename them.
STEMS_VETO_DISABLED = "stems_disabled"
STEMS_VETO_UNAVAILABLE = "stems_unavailable"
STEMS_VETO_OUTGOING_INSTRUMENTAL = "outgoing_instrumental"
STEMS_VETO_OUTGOING_VOCAL_DOMINANCE = "outgoing_vocal_dominance"
# Plan-time proxy for the planner's near-instrumental line
# (_INSTRUMENTAL_DUTY_MAX = 0.05 in smart_fades/planner/candidates.py): an
# outgoing deck under this mean vocal-duty density has nothing a DEFAULT_DUCK_DB
# (-6 dB, smart_fades/stems.py) duck could audibly remove.
STEMS_VETO_INSTRUMENTAL_DENSITY: float = 0.05

# Pair-level genre families for the auto-choice rules (research verdicts,
# 0.0.27): quick_cut is the natural default in hip-hop/pop/rock/funk but must
# never be an AUTO choice over four-on-the-floor material; double-drop is
# DnB-only (elsewhere it reads as mud); backspin is never chosen
# automatically. An explicit USER pick is never vetoed.
FOUR_ON_FLOOR_GENRES: tuple[str, ...] = (
    "house",
    "techno",
    "trance",
    "electro",
    "disco",
    "dance",
)
DNB_GENRES: tuple[str, ...] = ("drum and bass", "drum & bass", "jungle")

# Vocal-dense / radio-material genre family (pop, hip-hop, rock, ...): the
# third PAIR_BAR_CEILINGS family, hoisted to a name so the lane C phrase-math
# candidates share the exact same keyword list (one authority, two readers).
RADIO_GENRES: tuple[str, ...] = (
    "pop",
    "hip hop",
    "hip-hop",
    "rap",
    "rock",
    "punk",
    "metal",
    "indie",
    "folk",
    "soul",
    "funk",
)

# transition_bars stays a global control, but a pair's genre may clamp it
# (research: house/techno/trance run 32-bar blends, DnB tops out at 8,
# pop/hip-hop/rock/funk live at 4-8 with 16 as the hard cap). The clamp is
# advisory: it rides the transition report; the pinned automix map stays a
# (style, strictness) 2-tuple.
PAIR_BAR_CEILINGS: tuple[tuple[tuple[str, ...], int], ...] = (
    (FOUR_ON_FLOOR_GENRES, 32),
    (DNB_GENRES, 8),
    (RADIO_GENRES, 16),
)

# Per-transition derived-bar reason ids (0.0.28 lane C). Stable snake_case
# values the FE reads from the mix response's per-transition metadata
# ("transition_bars_reason"); they are API surface, never rename them.
TRANSITION_BARS_REASON_GLOBAL = "global_setting"
TRANSITION_BARS_REASON_FAMILY = "genre_family_phrase_math"
TRANSITION_BARS_REASON_CLIMB = "arc_climb_long_build"
TRANSITION_BARS_REASON_RESET = "arc_reset_short"
TRANSITION_BARS_REASON_VOCAL = "vocal_pair_short"
TRANSITION_BARS_REASON_OVERRIDE = "user_override"

# Phrase-math candidates per genre family (0.0.28 lane C): (short, long) are
# the family's clean phrase lengths, verified against the engine/planner
# usage sites and the PAIR_BAR_CEILINGS research verdicts - house/techno/
# trance run 32-bar blends (the long stop), DnB tops out at 8 (the build
# spec's raw {16, 32} would breach that verified ceiling and is NOT shipped),
# and pop/hip-hop/rock material lives at 4-8 with 16 as the hard cap. A pair
# spanning families takes the STRICTEST combination (min of both stops).
BAR_FAMILY_CANDIDATES: tuple[tuple[tuple[str, ...], tuple[int, int]], ...] = (
    (FOUR_ON_FLOOR_GENRES, (8, 32)),
    (DNB_GENRES, (4, 8)),
    (RADIO_GENRES, (4, 8)),
)


def _genre_of(track: dict[str, Any] | None) -> str:
    """Lowercased genre string of a track dict ("" when absent)."""
    if not track:
        return ""
    return str(track.get("genre") or "").lower()


def _pair_bar_cap(outgoing: dict[str, Any] | None, incoming: dict[str, Any]) -> int | None:
    """
    Return the strictest genre bar ceiling for either side of a pair.

    Returns None when neither side matches a profiled genre family.
    """
    caps: list[int] = []
    for genre in (_genre_of(outgoing), _genre_of(incoming)):
        if not genre:
            continue
        for keywords, ceiling in PAIR_BAR_CEILINGS:
            if any(keyword in genre for keyword in keywords):
                caps.append(ceiling)
                break
    return min(caps) if caps else None


def _bar_family_candidates(
    outgoing: dict[str, Any] | None, incoming: dict[str, Any]
) -> tuple[int, int] | None:
    """
    Return the strictest (short, long) phrase-math candidates for a pair.

    Either side's genre may match a family; a pair spanning families takes
    the min of both stops so the boundary satisfies both phrase maths.
    Returns None when neither side matches a profiled family - the user's
    global bars then pass through untouched.
    """
    genres = (_genre_of(outgoing), _genre_of(incoming))
    matched = [
        candidates
        for keywords, candidates in BAR_FAMILY_CANDIDATES
        if any(any(keyword in genre for keyword in keywords) for genre in genres if genre)
    ]
    if not matched:
        return None
    return (min(lo for lo, _ in matched), min(hi for _, hi in matched))


def _energy_of(track: dict[str, Any] | None) -> float | None:
    """Finite numeric energy of a track row (nested analysis loses to top level)."""
    if not track:
        return None
    merged = {**(track.get("analysis") or {}), **track}
    energy = merged.get("energy")
    if isinstance(energy, bool) or not isinstance(energy, (int, float)):
        return None
    value = float(energy)
    return value if math.isfinite(value) else None


def derive_transition_bars(
    outgoing: dict[str, Any] | None,
    incoming: dict[str, Any],
    control: DJControls,
) -> tuple[int, str]:
    """
    Derive the effective transition bar count for ONE boundary (0.0.28 lane C).

    Deterministic plan-time decision from data the planner already has - the
    flat global constant is only the BASE now: the pair's genre-family phrase
    math bounds it, the energy-arc position picks the family's long stop on a
    climb (longer builds) and the short stop on a reset (resets stay short),
    and a KNOWN vocal-vs-vocal boundary always falls back to the short stop
    (vocal material gets the short/safe handover even mid-climb). The user's
    per-track ``transition_bars_overrides`` are applied by the callers and
    always win over this derivation. Returns (bars, reason) with a stable
    snake_case reason id for the mix response metadata.
    """
    bars = control.transition_bars
    reason = TRANSITION_BARS_REASON_GLOBAL
    family = _bar_family_candidates(outgoing, incoming)
    if family is None:
        return bars, reason
    short, long = family
    clamped = max(short, min(bars, long))
    if clamped != bars:
        bars = clamped
        reason = TRANSITION_BARS_REASON_FAMILY
    cur_e = _energy_of(outgoing)
    cand_e = _energy_of(incoming)
    if cur_e is not None and cand_e is not None:
        delta = cand_e - cur_e
        if delta <= -ARC_RESET_DROP:
            bars = short
            reason = TRANSITION_BARS_REASON_RESET
        elif delta >= ARC_CLIMB_MAX:
            bars = long
            reason = TRANSITION_BARS_REASON_CLIMB
    # a KNOWN vocal-vs-vocal boundary (instrumental is False on both sides,
    # the same merged-row convention _transition_report uses) stays short:
    # unknown vocal status must not shorten a blend, a known one must.
    cur = {**(outgoing.get("analysis") or {}), **outgoing} if outgoing else {}
    cand = {**(incoming.get("analysis") or {}), **incoming}
    if cur.get("instrumental") is False and cand.get("instrumental") is False and bars > short:
        bars = short
        reason = TRANSITION_BARS_REASON_VOCAL
    return bars, reason


def _style_guard(
    style: str, outgoing: dict[str, Any] | None, incoming: dict[str, Any]
) -> list[str]:
    """
    Auto-choice style blocks (research verdicts, 0.0.27).

    Returns the stable rule ids for every style the pair must NOT fall into
    automatically: quick_cut over four-on-the-floor material, double-drop
    outside DnB, backspin anywhere. NEVER applied to an explicit user pick.
    """
    blocked: list[str] = []
    genres = (_genre_of(outgoing), _genre_of(incoming))
    four_on_floor = any(
        any(keyword in genre for keyword in FOUR_ON_FLOOR_GENRES) for genre in genres if genre
    )
    dnb = any(any(keyword in genre for keyword in DNB_GENRES) for genre in genres if genre)
    if style == "quick_cut" and four_on_floor:
        blocked.append(RULE_QUICK_CUT_BLOCKED)
    if style == "double_drop" and not dnb:
        blocked.append(RULE_DOUBLE_DROP_NON_DNB)
    if style == "backspin":
        blocked.append(RULE_BACKSPIN_NEVER_AUTO)
    return blocked


def _pair_style(
    outgoing: dict[str, Any] | None,
    incoming: dict[str, Any],
    control: DJControls,
) -> dict[str, Any]:
    """
    ONE pair-aware authority for genre -> (style, strictness) resolution.

    Both _transition_report and _automix_blend_prefs consult this so they can
    no longer disagree: with the user's style on "auto", the INCOMING track's
    genre fills the pin first and the OUTGOING track's second -- before this
    resolver only the incoming side was ever read, and the two call sites
    each read a different track. An explicit user pick always wins and is
    never vetoed by the style guards (they block AUTO choices only).

    Returns a dict: ``style``, ``strictness``, ``source`` (user | genre |
    genre_outgoing | engine), ``rules`` (stable snake_case ids for the FE's
    plain-language definitions) and ``bars`` (the genre-clamped effective
    transition bar count; ``rules`` carries transition_bars_capped when the
    clamp bit).
    """
    style = control.transition_style
    strictness = control.beat_strictness
    source = "user" if style != "auto" else None
    rules: list[str] = []
    if source is None:
        incoming_profile = _genre_profile(incoming.get("genre"))
        outgoing_profile = _genre_profile((outgoing or {}).get("genre"))
        profile = incoming_profile or outgoing_profile
        if profile is not None:
            style, strictness = profile
            source = "genre" if incoming_profile else "genre_outgoing"
        else:
            # style stays "auto": the planner decides per pair at blend time
            source = "engine"
        blocked = _style_guard(style, outgoing, incoming)
        if blocked:
            # Defensive: no current GENRE_PROFILES entry resolves to a
            # guarded style, but a future table row must never leak one
            # past here; fall back to the pair's own profile or the blend.
            rules.extend(blocked)
            fallback = incoming_profile or outgoing_profile
            style = fallback[0] if fallback else "full_blend"
    bars, bars_reason = derive_transition_bars(outgoing, incoming, control)
    cap = _pair_bar_cap(outgoing, incoming)
    if cap is not None and bars > cap:
        bars = cap
        rules.append(RULE_TRANSITION_BARS_CAPPED)
    elif bars_reason == TRANSITION_BARS_REASON_FAMILY:
        # the family phrase math lowered the user's base: same honest
        # "genre shortened your bars" signal as the cap, so the FE's
        # existing rule definition renders for it too
        rules.append(RULE_TRANSITION_BARS_CAPPED)
    return {
        "style": style,
        "strictness": strictness,
        "source": source,
        "rules": rules,
        "bars": bars,
        "bars_reason": bars_reason,
    }


def _stems_veto_reason(
    outgoing: dict[str, Any] | None,
    stems_enabled: bool,
    veto_override: bool = False,
) -> str | None:
    """
    Decide whether the stems vocal-duck may run for one transition.

    Per-transition veto (0.0.28 lane D): the overlap runs DRY (no stem
    processing) when separation is likely to hurt or helplessly noop, and the
    reason travels to the FE via the mix response's per-transition metadata.
    The thresholds are derived, not invented:

    - nothing to duck: the outgoing deck's mean vocal density sits under the
      planner's near-instrumental duty line (STEMS_VETO_INSTRUMENTAL_DENSITY,
      mirroring candidates._INSTRUMENTAL_DUTY_MAX), or the analysis flat-out
      claims instrumental - stems.duck_db_for_mask would scale the -6 dB depth
      to ~0 dB of actual duck, so processing is pure cost;
    - outgoing vocal dominance: the outgoing deck is at or past the index's
      vocal tier-2 line (quick_index.VOCAL_DENSE_DENSITY) - the duck is capped
      at stems.DEFAULT_DUCK_DB (-6 dB), so a dominant vocal stays
      half-suppressed over the incoming deck and the spectral gate leaves
      artifacts behind;
    - separation stack unavailable: no runtime/model, the duck could never
      apply (stems.available());

    and when the user's stems toggle is OFF the veto is reported as
    ``stems_disabled`` (default behavior: stems only run when toggled on).
    Returns None when the duck should run.
    """
    if not stems_enabled:
        return STEMS_VETO_DISABLED
    if veto_override:
        # the user's escape hatch past the threshold vetoes; the toggle
        # itself stays the master switch (an override never forces stems ON)
        return None
    if not stems_ducking.available():
        return STEMS_VETO_UNAVAILABLE
    if outgoing is None:
        return None
    # callers feed either nested rows ({"analysis": {...}}) or flat dicts;
    # analysis keys lose to the top level, same as _transition_report
    cur = {**(outgoing.get("analysis") or {}), **outgoing}
    density = cur.get("vocal_density")
    if isinstance(density, (int, float)) and not isinstance(density, bool):
        if density < STEMS_VETO_INSTRUMENTAL_DENSITY:
            return STEMS_VETO_OUTGOING_INSTRUMENTAL
        if density >= VOCAL_DENSE_DENSITY:
            return STEMS_VETO_OUTGOING_VOCAL_DOMINANCE
    elif cur.get("instrumental") is True:
        return STEMS_VETO_OUTGOING_INSTRUMENTAL
    return None


def _transition_report(
    ordered: list[dict[str, Any]],
    control: DJControls,
) -> list[dict[str, Any]]:
    """
    One self-explanation note per planned boundary, in playback order.

    Plan-time facts only: the renderer still picks the final rendered style
    per pair at blend time (its selection sees beat grids and vocal windows
    the plan does not). These are the inputs that decision reads: tempo
    delta, key affinity, vocal-collision risk, and which style/strictness
    the boundary is pinned to -- plus where that pin came from.
    """
    notes: list[dict[str, Any]] = []
    for current, candidate in pairwise(ordered):
        # callers feed either nested rows ({"analysis": {...}} from queue/
        # playlist snapshots) or flat dicts; analysis keys lose to the
        # top level so overrides stay authoritative
        cur = {**(current.get("analysis") or {}), **current}
        cand = {**(candidate.get("analysis") or {}), **candidate}
        a, b = cur.get("bpm"), cand.get("bpm")
        bpm_delta_pct = (
            round(abs(float(b) - float(a)) / float(a) * 100, 2)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a)
            else None
        )
        camelot_a, camelot_b = cur.get("camelot"), cand.get("camelot")
        key_distance = camelot_affinity(camelot_a, camelot_b) if camelot_a and camelot_b else None
        resolved = _pair_style(cur, cand, control)
        veto = _stems_veto_reason(current, control.stems_enabled, control.stems_veto_override)
        # the user's per-track override (0 = hard cut) always wins over the
        # derivation (0.0.28 lane C contract)
        override = control.transition_bars_overrides.get(str(cand.get("queue_item_id")))
        if override is not None:
            bars, bars_reason = override, TRANSITION_BARS_REASON_OVERRIDE
        else:
            bars, bars_reason = resolved["bars"], resolved["bars_reason"]
        notes.append(
            {
                "to": cand.get("name"),
                "bpm_delta_pct": bpm_delta_pct,
                "key_distance": key_distance,
                "both_vocal": cur.get("instrumental") is not True
                and cand.get("instrumental") is not True,
                "style": resolved["style"],
                "strictness": resolved["strictness"],
                "source": resolved["source"],
                # stable snake_case rule ids for the FE's plain-language
                # definitions (server half of the definitions contract)
                "rules": resolved["rules"],
                # the pair's derived effective transition length (lane C)
                # -- or the user's per-track override, which always wins
                "transition_bars": bars,
                # stable snake_case reason id for the derived value
                "transition_bars_reason": bars_reason,
                # per-transition stems veto (0.0.28 lane D): when separation is
                # likely to hurt (or would noop) the overlap runs dry and the
                # reason rides along; a null reason means the duck ran
                "stems_used": veto is None,
                "stems_veto_reason": veto,
            }
        )
    return notes


def _automix_blend_prefs(
    tracks: list[dict[str, Any]],
    control: DJControls,
    previous: dict[str, Any] | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Pin (transition_style, beat_strictness) for every AutoMix-pinned track.

    Same keys as the duration pins (the blend INTO each track reads its own
    pref). Style "auto" + strictness "normal" means engine defaults, which the
    streams-side getter collapses to no pin at all. With the style on "auto",
    the pair-aware _pair_style fills the pin: each track's OUTGOING
    predecessor (the previous entry in this list, or ``previous`` for the
    first entry) is consulted when the track's own genre has no profile --
    before 0.0.27 only the incoming side was ever read. An explicit user
    pick always wins.
    """
    prefs: dict[str, tuple[str, str]] = {}
    outgoing = previous
    for track in tracks:
        if track.get("queue_item_id") is None:
            continue
        resolved = _pair_style(outgoing, track, control)
        prefs[str(track["queue_item_id"])] = (resolved["style"], resolved["strictness"])
        outgoing = track
    return prefs


def _automix_stems_vetoes(
    tracks: list[dict[str, Any]],
    control: DJControls,
    previous: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Pin per-transition stems vetoes for AutoMix, same keys as the blend prefs.

    The blend INTO each track reads its own veto, computed against that
    track's OUTGOING predecessor (same walk as _automix_blend_prefs).
    "stems_disabled" is deliberately omitted: with the user's toggle off the
    global stems pin already keeps every overlap dry.
    """
    vetoes: dict[str, str] = {}
    outgoing = previous
    for track in tracks:
        if track.get("queue_item_id") is None:
            continue
        reason = _stems_veto_reason(outgoing, control.stems_enabled, control.stems_veto_override)
        if reason is not None and reason != STEMS_VETO_DISABLED:
            vetoes[str(track["queue_item_id"])] = reason
        outgoing = track
    return vetoes


def _camelot_from_key(key: str | None, mode: str | None) -> str | None:
    """Convert Music Assistant key/mode to Camelot notation."""
    # one authority since 0.0.27: the quick-pull index derives its key buckets
    # from the SAME mapping (quick_index.camelot_from_key), so the planner and
    # the index can never disagree on a track's Camelot code.
    return quick_index.camelot_from_key(key, mode)


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


PULL_CANDIDATES_MAX = 250
PULL_EXCLUDE_MAX = 500


def _pull_filters(  # noqa: PLR0913 - filter surface mirrors the command's
    *,
    limit: int = 25,
    bpm: float | None = None,
    bpm_tolerance: float = 0.06,
    key: str | None = None,
    energy: float | None = None,
    energy_band: str | None = None,
    mood: str | None = None,
    era: str | None = None,
    year: int | None = None,
    vocal: int | str | None = None,
    exclude_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate pull_candidates filters into quick_index.select kwargs.

    Raises ValueError naming the offending filter; the command translates
    that into its named SmartDJError surface. Untrusted JSON scalars must
    never reach the query builder unchecked.
    """
    out: dict[str, Any] = {}
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"limit={limit!r}")  # noqa: TRY004 - translated to SmartDJError
    out["limit"] = max(1, min(limit, PULL_CANDIDATES_MAX))
    if bpm is not None:
        value = _finite_in(bpm, positive=True)
        if value is None:
            raise ValueError(f"bpm={bpm!r}")
        out["bpm"] = value
        tol = _finite_in(bpm_tolerance, 0.01, 0.5)
        out["bpm_tolerance"] = 0.06 if tol is None else tol
    if key is not None:
        if not isinstance(key, str) or not compatible_buckets(key.strip().upper()):
            raise ValueError(f"key={key!r}")
        out["key"] = key.strip().upper()
    if energy is not None:
        value = _finite_in(energy, 0.0, 1.0)
        if value is None:
            raise ValueError(f"energy={energy!r}")
        out["energy"] = value
    if energy_band is not None:
        if energy_band not in ENERGY_BANDS:
            raise ValueError(f"energy_band={energy_band!r}")
        out["energy_band"] = energy_band
    if mood is not None:
        if not isinstance(mood, str) or not mood.strip():
            raise ValueError(f"mood={mood!r}")
        out["mood"] = mood.strip().lower()
    if era is not None:
        if not isinstance(era, str) or not re.fullmatch(r"\d{4}s", era.strip().lower()):
            raise ValueError(f"era={era!r}")
        out["era"] = era.strip().lower()
    if year is not None:
        if isinstance(year, bool) or not isinstance(year, int) or not 1000 <= year <= 3000:
            raise ValueError(f"year={year!r}")
        out["year"] = year
    if vocal is not None:
        if isinstance(vocal, bool):
            raise ValueError(f"vocal={vocal!r}")
        if isinstance(vocal, int) and vocal in VOCAL_TIER_NAMES:
            out["vocal"] = vocal
        elif isinstance(vocal, str) and vocal.strip().lower() in VOCAL_TIER_NAMES.values():
            out["vocal"] = vocal.strip().lower()
        else:
            raise ValueError(f"vocal={vocal!r}")
    if exclude_ids is not None:
        if not isinstance(exclude_ids, list):
            raise ValueError(f"exclude_ids={exclude_ids!r}")
        out["exclude_ids"] = [str(item) for item in exclude_ids][:PULL_EXCLUDE_MAX]
    return out


async def _fetch_radio_tracks(
    request: Any,
    seed: dict[str, Any],
    count: int,
) -> list[Any]:
    """
    Fetch Musicae radio seeds (primary path 2026 docs, flat variant once).

    Auth/quota failures surface as MusicaeRateLimited with their identity
    intact; only shape/transport problems are aggregated into one error.
    """
    limit = max(1, min(int(count), 50))
    errors = []
    data = None
    for path, params in (
        (f"/v2/tracks/{seed['item_id']}/radio", {"limit": limit}),
        ("/v2/radio", {"seed_id": seed["item_id"], "limit": limit}),
    ):
        try:
            data = await request(path, params)
            break
        except MusicaeRateLimited:
            raise  # auth/quota failures keep their identity; do not flatten
        except RuntimeError as err:
            errors.append(f"{path}: {err}")
    if not data:
        raise RuntimeError("Musicae radio failed - " + " | ".join(errors))
    raw_tracks = data.get("tracks") if isinstance(data, dict) else None
    if not isinstance(raw_tracks, list):
        raise RuntimeError(  # noqa: TRY004 - vendor shape, not a type error
            f"Unexpected Musicae radio response shape: {str(raw_tracks)[:300]}"
        )
    return raw_tracks


async def _playlist_analysis_rows(
    provider: Any,
    resolved: list[Any],
    analysis_provider: str,
    pool_source: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Analyze resolved playlist tracks; shared by mix_playlist + save_sorted.

    Returns (analyzed rows with mix- ids and nested analysis, media_by_id
    mapping mix- id -> the resolved media item). Analysis failures are kept
    as rows with analysis=None; the optimizer scores them last. Per-track
    analyses resolve through the provider's quick-pull index first
    (``_pool_analyses`` fast path), falling back to the full analysis path.
    """
    analysis_rows: list[dict[str, Any]] = []
    media_by_id: dict[str, Any] = {}
    for idx, track in enumerate(resolved):
        metadata = getattr(track, "metadata", None)
        genres = list(getattr(metadata, "genres", None) or [])
        mix_id = f"mix-{idx}"
        # analysis keys resolve through provider mappings (review finding,
        # Oct 24): a library playlist's rows used to carry provider="library",
        # which no analysis store or index can answer - every row planned as
        # "analysis unavailable" and the engine degraded to fixed barriers
        targets = provider._analysis_targets_for_track(track)
        a_item_id, a_provider = (
            targets[0] if targets else (track.item_id, getattr(track, "provider", None))
        )
        analysis_rows.append(
            {
                "queue_item_id": mix_id,
                "name": track.name,
                "artist": getattr(track, "artist_str", None) or getattr(track, "artist", None),
                "item_id": a_item_id,
                "provider": a_provider,
                "genre": genres[0] if genres else None,
                "genres": genres,
                "explicit": getattr(metadata, "explicit", None) if metadata else None,
            }
        )
        media_by_id[mix_id] = track

    analyses = await provider._pool_analyses(analysis_rows, analysis_provider, pool_source)
    analyzed: list[dict[str, Any]] = []
    for item, result in zip(analysis_rows, analyses, strict=True):
        analysis = None if isinstance(result, Exception) else result
        if isinstance(result, Exception):
            provider.logger.debug("Smart DJ analysis failed for %s: %s", item["item_id"], result)
        analyzed.append({**item, "analysis": analysis})
    return analyzed, media_by_id


class SmartDJProvider(PluginProvider):
    """Native Smart DJ controller and Musicae enrichment client."""

    async def analyze(
        self,
        queue_id: str,
        limit: int = 40,
        analysis_provider: str = "auto",
        pool_source: str = "auto",
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

        analyses = await self._pool_analyses(items, analysis_provider, pool_source)
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
                # the sweep's hard timeout, applied to the preroll path: a
                # hung stream fetch here holds the same per-item
                # _stream_details_locks entry the PLAYBACK path needs, so an
                # unbounded preroll can stall a real transition
                done = await asyncio.wait_for(
                    self._preroll_track_analysis(item),
                    timeout=ANALYSIS_ATTEMPT_TIMEOUT,
                )
            except TimeoutError:
                self.logger.warning(
                    "Smart DJ preroll timed out for %s after %ss",
                    item.name,
                    ANALYSIS_ATTEMPT_TIMEOUT,
                )
                failed[str(item.queue_item_id)] = (
                    f"timeout after {ANALYSIS_ATTEMPT_TIMEOUT}s"
                )
                continue
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
        # the controls param was previously dead (audit note); it now feeds
        # the transition report so the notes reflect the caller's pins
        control = self._parse_controls(dict(controls or {}), 0.08, True, True, "ai_dj")

        # Musicae radio: primary path per the 2026 docs, seeded by one track id.
        # The endpoint shape is path-parameterized (/v2/tracks/{id}/radio); if
        # that 404s, try the flat variant once. A wrong path raises with both
        # response bodies so the shape is visible in the error.
        raw_tracks = await _fetch_radio_tracks(self._request, seed, count)

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
            if len(tracks) >= max(1, min(int(count), 50)):
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
        # _transition_report reads every element with .get(); the appended
        # tracks here are live media objects (not snapshot dicts), so flatten
        # them into the same row shape the other call sites feed. Without this
        # the report raised AttributeError AFTER the queue was already mutated
        # and the preroll fired, so the FE saw "Complete mix failed" while the
        # music had in fact changed.
        report_rows = [
            seed,
            *(
                {
                    "name": track.name,
                    "artist": getattr(track, "artist_str", None) or getattr(track, "artist", None),
                    "queue_item_id": queue_item.queue_item_id,
                    "analysis": None,
                }
                for track, queue_item in zip(tracks, queue_items, strict=True)
            ),
        ]
        return {
            "queue_id": queue_id,
            "seed": seed.get("name"),
            "added": [track.name for track in tracks],
            "automix_pinned": automix_pinned,
            "transition_notes": _transition_report(report_rows, control),
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
            # resolve through provider mappings (review finding, Oct 24):
            # library tracks' own ids produced `library:` cache keys nothing
            # ever reads - up to 500 wasted Musicae requests per run
            for a_item_id, a_instance in self._analysis_targets_for_track(track):
                candidate = f"{a_instance}:{a_item_id}"
                cached = self._cache.get(candidate)
                if not (cached and now - cached[0] < CACHE_TTL):
                    key = (candidate, a_item_id)
                    break
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
            except MusicaeRateLimited:
                quota_exhausted = True
                break
            except RuntimeError as err:
                if "429" in str(err):
                    # fallback for raw vendor 429s that never became typed
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

    async def mix_playlist(
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
        # candidate-pool source (0.0.27 Item 3): "auto" pulls per-track
        # analyses from the derived quick-pull index when fresh (fast path),
        # "full" forces the pre-index payload path
        pool_source = str(raw.get("pool_source", "auto"))
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

        # Shared mix pipeline (0.0.27): analysis -> plan -> queue apply ->
        # automix pins -> response. mix_playlist's response keys are carried
        # verbatim via response_base; mix_tracks ends in the same helper.
        return await self._mix_resolved(
            queue_id=queue_id,
            resolved=resolved,
            control=control,
            mode=mode,
            bpm_tolerance=bpm_tolerance,
            analysis_provider=analysis_provider,
            pool_source=pool_source,
            response_base={
                "playlist_provider": playlist_provider,
                "playlist_item_id": playlist_item_id,
                "playlist_total": playlist_total,
                "truncated": truncated,
            },
        )

    async def mix_tracks(
        self,
        queue_id: str,
        track_uris: list[str] | None = None,
        uris: list[str] | None = None,
        bpm_tolerance: float = 0.08,
        prefer_keys: bool = True,
        preserve_variety: bool = True,
        mode: str = "ai_dj",
        controls: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Mix an explicit list of track URIs onto the queue in Smart DJ order.

        0.0.27: the FE library browser hands over a selection directly
        instead of building a scratch playlist (which litters the user's
        library). Each URI resolves through the standard item-by-uri path;
        unresolvable URIs are skipped and reported in the response. Shares
        the whole analyze -> plan -> apply -> pin pipeline with mix_playlist
        via _mix_resolved.
        """
        raw = dict(controls or {})
        analysis_provider = str(raw.get("analysis_provider", "auto"))
        pool_source = str(raw.get("pool_source", "auto"))
        control = self._parse_controls(raw, bpm_tolerance, prefer_keys, preserve_variety, mode)
        # `uris` is accepted as an alias: the FE's mixSelection shipped the
        # payload key `uris` against this method's `track_uris` (review
        # finding, Oct 24) and every such call failed on binding
        payload = [str(uri) for uri in (track_uris or uris or [])]
        resolved, unresolved = await self._resolve_track_uris(payload)
        if len(resolved) < 2:
            raise RuntimeError(
                f"Need at least 2 resolvable tracks to mix ({len(resolved)} of {len(payload)} URIs)"
            )
        return await self._mix_resolved(
            queue_id=queue_id,
            resolved=resolved,
            control=control,
            mode=mode,
            bpm_tolerance=bpm_tolerance,
            analysis_provider=analysis_provider,
            pool_source=pool_source,
            response_base={"track_uris": len(payload), "unresolved_uris": unresolved},
        )

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
        # candidate-pool source: "auto" pulls from the derived quick-pull
        # index when fresh (fast path), "full" forces the payload path
        pool_source = str(raw.get("pool_source", "auto"))
        snapshot = await self.analyze(
            queue_id, analysis_provider=analysis_provider, pool_source=pool_source
        )
        tracks = snapshot["tracks"]
        if len(tracks) < 2:
            # Keep the rank contract shape even when there is nothing to rank:
            # the FE reads applied/queue_changed/automix on every response and
            # the raw analyze payload carries none of them.
            return {
                "queue_id": queue_id,
                "current_item_id": tracks[0]["queue_item_id"] if tracks else None,
                "tracks": [],
                "optimized_window": len(tracks),
                "queue_item_count": snapshot.get("queue_item_count"),
                "applied": False,
                "queue_changed": False,
                "automix_applied": False,
                "automix": None,
                "reason": "not_enough_tracks",
                "settings": {"controls": raw, "mode": mode},
            }

        control = self._parse_controls(raw, bpm_tolerance, prefer_keys, preserve_variety, mode)
        automix_pin: dict[str, Any] | None = None
        if not control.smart_reorder_enabled:
            if apply and control.automix_enabled:
                # Real AutoMix execution: the queue-level crossfade toggle is a
                # supported runtime command, so flipping it is safe even when no
                # reorder is requested.
                self.mass.player_queues.set_crossfade(queue_id, True)
                self.mass.streams.audio.set_automix_durations(
                    _automix_durations(tracks[1:], control, previous=tracks[0])
                )
                self.mass.streams.audio.set_automix_blend_prefs(
                    _automix_blend_prefs(tracks[1:], control, previous=tracks[0])
                )
                self.mass.streams.audio.set_automix_stems_vetoes(
                    _automix_stems_vetoes(tracks[1:], control, previous=tracks[0])
                )
                self.mass.streams.audio.set_automix_stems(control.stems_enabled)
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
        try:
            optimized = beam_optimize(
                tracks[1:],
                _merge_track(tracks[0]) if tracks else snapshot["current"],
                selected_mode,
                controls=control,
                beam_width=max(4, min(96, control.lookahead * 3)),
            )
        except HardRequirementsImpossible as err:
            # fail-soft plan contract (review finding, Oct 24): the FE's
            # plan() branches on plan_failed/hard_violations, which this
            # command never emitted - the insight panel waited on a shape
            # that did not exist. Rows are mapped into the FE's
            # HardViolation interface (suggestion.setting/.to).
            return {
                "queue_id": queue_id,
                "current_item_id": tracks[0]["queue_item_id"],
                "tracks": tracks[1:],
                "optimized_window": len(tracks) - 1,
                "queue_item_count": snapshot.get("queue_item_count"),
                "applied": False,
                "queue_changed": False,
                "automix_applied": False,
                "automix": None,
                "plan_failed": True,
                "hard_violations": [
                    {
                        "rule": row.get("rule"),
                        "message": None,
                        "actual": row.get("actual"),
                        "limit": row.get("limit"),
                        "unit": row.get("unit"),
                        "overshoot": None,
                        "severity": None,
                        "suggestion": (
                            {
                                "setting": row.get("suggested_setting"),
                                "label": None,
                                "from": None,
                                "to": row.get("suggested_value"),
                            }
                            if row.get("suggested_setting")
                            else None
                        ),
                    }
                    for row in (err.details or [])
                ],
                "candidates_rejected": 0,
                "reason": "hard_requirements_impossible",
                "settings": {"controls": raw, "mode": mode},
            }
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
                self.mass.streams.audio.set_automix_stems_vetoes({})
                self.mass.streams.audio.set_automix_stems(False)
            if control.automix_enabled:
                # Real AutoMix execution: hand the queue to the streams engine's
                # crossfade path (smart fades when analysis is available, standard
                # otherwise). A plan with automix disabled never touches the toggle.
                self.mass.player_queues.set_crossfade(queue_id, True)
                self.mass.streams.audio.set_automix_durations(
                    _automix_durations(optimized, control, previous=tracks[0])
                )
                self.mass.streams.audio.set_automix_blend_prefs(
                    _automix_blend_prefs(optimized, control)
                )
                self.mass.streams.audio.set_automix_stems_vetoes(
                    _automix_stems_vetoes(optimized, control)
                )
                self.mass.streams.audio.set_automix_stems(control.stems_enabled)
                self.mass.create_task(self.preroll_queue_analysis(queue_id, 10))
                automix_info = {
                    "enabled": True,
                    "mode": "smart" if self.mass.streams.smart_fades_available else "standard",
                }
        return {
            "queue_id": queue_id,
            "current_item_id": tracks[0]["queue_item_id"],
            "tracks": optimized,
            "transition_notes": _transition_report([tracks[0], *optimized], control),
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
                "transition_styles": [
                    "auto",
                    "full_blend",
                    "energy",
                    "quick_cut",
                    "radio",
                    "echo_out",
                    "filter_out",
                ],
                "beat_strictness": ["strict", "normal", "aggressive"],
                # 0.0.27: floating energy-arc control + direct track-uri mixes
                "energy_arc": True,
                "mix_tracks": True,
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
            ConfigEntry(
                key=CONF_PREANALYSIS_ENABLED,
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
                required=False,
                requires_reload=False,
                description=(
                    "Analyze newly added library tracks in the background, using "
                    "the same machinery and safeguards as the library sweep."
                ),
            ),
        )


    async def status(self) -> dict[str, Any]:
        """Return provider readiness without revealing credentials."""
        return {
            "configured": bool(self._key()),
            "musicae_host": MUSICAE_HOST,
            "cached_tracks": len(self._cache),
            # quick-pull index progress (0.0.28 FE contract): indexed rows vs
            # total library tracks. Both fields are OPTIONAL-friendly: absent
            # or None reads as "unknown" on older/partial states, never an
            # error.
            "total_indexed": await self._status_total_indexed(),
            "library_tracks": await self._status_library_tracks(),
        }

    async def pull_candidates(  # noqa: PLR0913 - filter surface mirrors spec 3.3
        self,
        limit: int = 25,
        bpm: float | None = None,
        bpm_tolerance: float = 0.06,
        key: str | None = None,
        energy: float | None = None,
        energy_band: str | None = None,
        mood: str | None = None,
        era: str | None = None,
        year: int | None = None,
        vocal: int | str | None = None,
        exclude_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Query the derived quick-pull index for candidates (0.0.27 spec 3.3).

        A pure query surface: no audio scan, no analyzer quota, no cache
        writes. Filters (all optional, ANDed):
          - bpm + bpm_tolerance: tempo window on the normalized 80-160 bpm
            plus its 2:1 half/double aliases (87 <-> 174, 70 <-> 140);
          - key: a Camelot code ("8A"); candidates expand to the compatible
            buckets (the code itself, +/-1 on the same ring, same number on
            the opposite ring - matching engine.camelot_affinity's tiers);
          - energy: a 0-1 target matched as a +/-1 tier window, or
            energy_band: an exact band name (quick_index.ENERGY_BANDS);
          - mood: a quadrant tier (euphoric/peaceful/tense/melancholic) or a
            raw MA metadata mood string (models metadata.py:191);
          - era ("1980s") or year (1987) from release metadata;
          - vocal: tier 0/1/2 or name (instrumental/balanced/vocal);
          - exclude_ids: track ids to skip (recently played), capped at 500.
        Returns matching (item_id, provider_instance) pairs with the derived
        values they matched on. Malformed filters raise SmartDJError
        ("invalid_control"); an unindexed library simply returns [].
        """
        try:
            filters = _pull_filters(
                limit=limit,
                bpm=bpm,
                bpm_tolerance=bpm_tolerance,
                key=key,
                energy=energy,
                energy_band=energy_band,
                mood=mood,
                era=era,
                year=year,
                vocal=vocal,
                exclude_ids=exclude_ids,
            )
        except ValueError as err:
            raise SmartDJError("invalid_control", str(err)) from None
        index = getattr(self, "_quick_index", None)
        if index is None:
            raise SmartDJError("index_unavailable", "quick-pull index not initialized")
        try:
            await asyncio.to_thread(index.open)
            rows = await asyncio.to_thread(index.select, **filters)
            total = await asyncio.to_thread(index.count)
        except (OSError, ValueError) as err:
            # a corrupt/unreadable index must not take the command down;
            # an honest empty answer plus a log line, and the backfill can
            # rebuild on the next load
            self.logger.warning("Smart DJ quick-pull index query failed: %s", err)
            rows, total = [], 0
        return {
            "candidates": rows,
            "count": len(rows),
            "total_indexed": total,
            "index_version": INDEX_VERSION,
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
        # Derived quick-pull index (0.0.27 Item 3): one derived row per
        # analyzed track, in the provider's OWN storage dir - never inside
        # audio_analysis.db (version-checked, shared surface, spec 3.1).
        self._quick_index = QuickIndex(Path(self.mass.storage_path) / "smart_dj" / INDEX_FILENAME)
        self._index_backfill_task: asyncio.Task[int] | None = None
        # UI injection state: the patched index.html and the restored original.
        self._tmp_index_path: str | None = None
        self._orig_index_path: str | None = None
        self._unregister_ui_route: Any = None
        # Library-wide local analysis sweep state (smart_dj/analyze_library_local).
        self._library_scan_task: asyncio.Task[None] | None = None
        self._library_scan_cancel = asyncio.Event()
        self._library_scan_state: dict[str, Any] = {}
        # Pre-analysis on arrival (0.0.28 lane B): bounded queue of newly
        # added library tracks pending background analysis, drained by a
        # single worker that reuses the sweep's machinery and safeguards.
        self._preanalysis_queue: deque[Any] = deque()
        self._preanalysis_pending_uris: set[str | None] = set()
        self._preanalysis_task: asyncio.Task[None] | None = None
        self._preanalysis_deferring = False

    async def loaded_in_mass(self) -> None:
        """Register the Smart DJ API commands and load the persistent cache."""
        await self._load_cache()
        # the quick-pull index must exist before any command can query it;
        # open is idempotent and the backfill only ADDS missing rows
        await asyncio.to_thread(self._quick_index.open)
        self._index_backfill_task = self.mass.create_task(self._quick_index_backfill())
        # Pre-analysis on arrival (0.0.28 lane B): a MEDIA_ITEM_ADDED event for
        # a track enqueues the item into the bounded pre-analysis queue; the
        # worker reuses the sweep's machinery (same hard timeout, same
        # concurrency limiter size, same deferral, same failure accounting,
        # same _index_upsert success point). The returned unsubscribe handle
        # is stored with the API handles so unload drops the listener too.
        self._handles.append(
            self.mass.subscribe(self._on_media_added, event_filter=EventType.MEDIA_ITEM_ADDED)
        )
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
            ("smart_dj/mix_tracks", self.mix_tracks, Scope.LIBRARY_WRITE),
            ("smart_dj/save_sorted_playlist", self.save_sorted_playlist, Scope.LIBRARY_WRITE),
            ("smart_dj/status", self.status, None),
            ("smart_dj/capabilities", self.capabilities, None),
            (
                "smart_dj/preroll_queue_analysis",
                self.preroll_queue_analysis,
                Scope.LIBRARY_WRITE,
            ),
            ("smart_dj/complete_mix", self.complete_mix, Scope.LIBRARY_WRITE),
            ("smart_dj/analyze_library", self.analyze_library, Scope.LIBRARY_WRITE),
            ("smart_dj/analyze_playlist", self.analyze_playlist, Scope.LIBRARY_WRITE),
            ("smart_dj/analyze_library_local", self.analyze_library_local, Scope.LIBRARY_WRITE),
            (
                "smart_dj/create_analyzed_playlist",
                self.create_analyzed_playlist,
                Scope.LIBRARY_WRITE,
            ),
            (
                "smart_dj/pull_candidates",
                self.pull_candidates,
                Scope.LIBRARY_WRITE,
            ),
        )
        for command, handler, required_scope in handlers:
            self._handles.append(
                self.mass.register_api_command(command, handler, required_scope=required_scope)
            )
        self._setup_ui()

    async def unload(self, is_removed: bool = False) -> None:
        """Close the HTTP client, drop the UI, and unregister commands."""
        pre_task = self._preanalysis_task
        if pre_task is not None and not pre_task.done():
            pre_task.cancel()
            with suppress(asyncio.CancelledError):
                await pre_task
        self._preanalysis_task = None
        self._preanalysis_queue.clear()
        self._preanalysis_pending_uris.clear()
        scan_task = self._library_scan_task
        if scan_task is not None and not scan_task.done():
            self._library_scan_cancel.set()
            scan_task.cancel()
            with suppress(asyncio.CancelledError):
                await scan_task
        self._library_scan_task = None
        backfill_task = self._index_backfill_task
        if backfill_task is not None and not backfill_task.done():
            backfill_task.cancel()
            with suppress(asyncio.CancelledError):
                await backfill_task
        self._index_backfill_task = None
        self._teardown_ui()
        for handle in self._handles:
            handle()
        self._handles.clear()
        if self._session:
            await self._session.close()
            self._session = None
        await super().unload(is_removed)

    async def analyze_playlist(
        self,
        playlist_item_id: str,
        playlist_provider: str,
        limit: int = 500,
    ) -> dict[str, Any]:
        """
        Analyze a playlist's unanalyzed tracks in the background.

        Each track is streamed silently through the LOCAL ML analysis providers
        (no playback, no audible output, audio never stored) and the result is
        persisted in the analysis database, so the Smart DJ optimizer and the
        smart-fades planner have beat grids and RMS to work with. No Musicae
        quota is consumed. Concurrency follows the streams background-scan
        setting (default 2, clamped here to 1-8).
        """
        if not self.mass.streams.audio_analysis.providers:
            raise RuntimeError("No audio analysis providers are loaded")
        cap = max(1, min(int(limit), 2000))
        tracks: list[Any] = [
            item
            async for item in self.mass.music.playlists.tracks(playlist_item_id, playlist_provider)
        ][:cap]
        # resolve lightweight item mappings to full tracks (provider/item ids usable)
        resolved: list[Any] = []
        for track in tracks:
            if getattr(track, "metadata", None) is None and getattr(track, "item_id", None):
                with suppress(Exception):
                    track = await self.mass.music.tracks.get(  # noqa: PLW2901
                        track.item_id, track.provider
                    )
            if getattr(track, "item_id", None) and getattr(track, "provider", None):
                resolved.append(track)
        concurrency = self._background_concurrency()

        sem = asyncio.Semaphore(concurrency)
        analyzed: list[str] = []
        skipped: list[str] = []
        failed: dict[str, str] = {}

        async def _one(track: Any) -> None:
            async with sem:
                try:
                    done = False
                    for a_item_id, a_instance in self._analysis_targets_for_track(track):
                        if await self._analyze_media_silent(a_item_id, a_instance, track=track):
                            done = True
                            break
                except Exception as err:
                    failed[str(track.item_id)] = str(err)
                    return
                if done:
                    analyzed.append(str(track.item_id))
                else:
                    skipped.append(str(track.item_id))

        await asyncio.gather(*(_one(t) for t in resolved))
        return {
            "playlist_item_id": playlist_item_id,
            "total": len(resolved),
            "analyzed": analyzed,
            "skipped": skipped,
            "failed": failed,
            "concurrency": concurrency,
        }

    async def analyze_library_local(
        self,
        command: str = "start",
        pause_while_playing: bool = True,
        limit: int = 0,
    ) -> dict[str, Any]:
        """
        Start, poll, or cancel the library-wide LOCAL analysis sweep.

        Streams every unanalyzed library track silently through the local ML
        analysis providers (no playback, no audible output, audio never
        stored, no Musicae quota) so the Smart DJ optimizer and the smart-fades
        planner have beat grids and RMS for the whole library.

        Safeguards:
        - ``command="start"`` is idempotent: a second start while a sweep is
          running just returns the live progress.
        - ``pause_while_playing=True`` (the default) defers ALL dispatch while
          any player is PLAYING - the local analyzer is CPU-hungry and must
          never compete with live audio decode.
        - concurrency follows the streams background-scan setting (1-8,
          default 2); each per-track attempt has a hard
          ``ANALYSIS_ATTEMPT_TIMEOUT`` (45s) and its buffer is always torn
          down - a timeout counts as a failure (breaker, errors,
          ``recent_failures``).
        - a circuit breaker aborts the sweep after
          ``LIBRARY_SCAN_MAX_FAILURES`` consecutive failures instead of
          hammering a broken provider.
        - ``command="cancel"`` stops the sweep between tracks; already-stored
          analysis is kept, so a re-run resumes effectively for free.

        ``limit`` caps how many library tracks the sweep considers (0 = up to
        ``LIBRARY_SCAN_MAX_TRACKS``).
        """
        if command not in {"start", "status", "cancel"}:
            raise RuntimeError("command must be one of start, status, cancel")
        if command == "status":
            return dict(self._library_scan_state)
        if command == "cancel":
            self._library_scan_cancel.set()
            task = self._library_scan_task
            if task is not None and not task.done():
                # cancelling the worker task interrupts any in-flight track
                # too; _analyze_media_silent's finally always tears the buffer
                # down, so an interrupted track leaves nothing half-done
                self._library_scan_state["cancelled"] = True
                task.cancel()
            return dict(self._library_scan_state)
        if self._library_scan_task is not None and not self._library_scan_task.done():
            return dict(self._library_scan_state)
        if not self.mass.streams.audio_analysis.providers:
            raise RuntimeError("No audio analysis providers are loaded")
        cap = (
            LIBRARY_SCAN_MAX_TRACKS
            if int(limit) <= 0
            else max(1, min(int(limit), LIBRARY_SCAN_MAX_TRACKS))
        )
        library_tracks = await self.mass.music.tracks.library_items(
            favorite=None,
            search=None,
            limit=cap,
            offset=0,
            order_by="sort_name",
            summary=False,
        )
        targets: list[Any] = [
            item
            for item in library_tracks
            if getattr(item, "item_id", None) and getattr(item, "provider", None)
        ]
        self._library_scan_cancel.clear()
        self._library_scan_state = {
            "running": True,
            "total": len(targets),
            "analyzed": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
            "recent_failures": [],
            "deferring": False,
            "cancelled": False,
            "started_at": time.time(),
            "finished_at": None,
        }
        self._library_scan_task = asyncio.create_task(
            self._library_scan_worker(targets, self._library_scan_state, pause_while_playing)
        )
        return dict(self._library_scan_state)

    async def create_analyzed_playlist(
        self,
        name: str = "Smart DJ - analyzed",
        provider_instance_id: str | None = None,
        search: str = "",
        limit: int = 0,
    ) -> dict[str, Any]:
        """
        Create a playlist from every library track that HAS analysis data.

        The result is a ready-to-mix pool: every track in it has the local ML
        analysis (bpm/key/rms) the Smart DJ optimizer and the smart-fades
        planner need, so mixing from this playlist always works at full
        quality. ``search`` narrows the pool to library tracks matching the
        search string (name/artist/album, same semantics as the MA search).
        Tracks are scanned in library order (sort_name); ``limit`` caps the
        scan (0 = up to ``LIBRARY_SCAN_MAX_TRACKS``). The playlist is created
        on ``provider_instance_id`` when given, else the builtin provider.
        """
        cap = (
            LIBRARY_SCAN_MAX_TRACKS
            if int(limit) <= 0
            else max(1, min(int(limit), LIBRARY_SCAN_MAX_TRACKS))
        )
        library_tracks = await self.mass.music.tracks.library_items(
            favorite=None,
            search=str(search) if search else None,
            limit=cap,
            offset=0,
            order_by="sort_name",
            summary=False,
        )
        uris: list[str] = []
        scanned = 0
        for item in library_tracks:
            item_id = getattr(item, "item_id", None)
            provider_id = getattr(item, "provider", None)
            if not item_id or not provider_id:
                continue
            scanned += 1
            found = False
            for a_item_id, a_instance in self._analysis_targets_for_track(item):
                existing = await self.mass.streams.audio_analysis.get_audio_analysis(
                    item_id=a_item_id,
                    provider_instance_id_or_domain=a_instance,
                )
                # has_real_analysis, not `is not None`: acoustid no-match rows are
                # all-null tombstones that would otherwise count as analyzed.
                if has_real_analysis(existing):
                    found = True
                    break
            if not found:
                continue
            uri = getattr(item, "uri", None)
            if uri:
                uris.append(uri)
        if not uris:
            raise RuntimeError(
                "No analyzed tracks found - run smart_dj/analyze_library_local first"
            )
        playlist = await self.mass.music.playlists.create_playlist(
            name=name, provider_instance_or_domain=provider_instance_id
        )
        # the internal handler runs inline: the action returns once the tracks
        # are actually in the playlist, not when a background task was queued
        await self.mass.music.playlists._handle_add_playlist_tracks(playlist.item_id, uris, None)
        return {
            "playlist_id": str(playlist.item_id),
            "name": name,
            "tracks": len(uris),
            "scanned": scanned,
        }

    async def save_sorted_playlist(
        self,
        playlist_provider: str,
        playlist_item_id: str,
        bpm_tolerance: float = 0.08,
        prefer_keys: bool = True,
        preserve_variety: bool = True,
        mode: str = "ai_dj",
        controls: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Sort a playlist into Smart DJ order and SAVE the order back.

        The playlist's own track list becomes the sort's persistence: the
        optimized order is written back through the provider, so every
        client sees it. Read-only playlists fail loud (the controller's
        editability + provider-feature checks run inside the handlers).
        The response carries the new order, per-track energy levels, and
        per-boundary transition notes.
        """
        raw = dict(controls or {})
        analysis_provider = str(raw.get("analysis_provider", "auto"))
        control = self._parse_controls(raw, bpm_tolerance, prefer_keys, preserve_variety, mode)
        playlist_tracks = [
            item
            async for item in self.mass.music.playlists.tracks(playlist_item_id, playlist_provider)
        ]
        if len(playlist_tracks) < 2:
            raise RuntimeError("Playlist needs at least 2 tracks to sort")
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
            raise RuntimeError("Playlist tracks could not be resolved for sorting")
        analyzed, media_by_id = await _playlist_analysis_rows(self, resolved, analysis_provider)
        preset = MODES.get(mode, MODES["ai_dj"])
        selected_mode = DJMode(
            preset.name,
            _bpm_tolerance(bpm_tolerance),
            preset.energy_direction,
            preset.variety,
            preset.weights,
        )
        optimized = beam_optimize(
            analyzed[1:],
            _merge_track(analyzed[0]),
            selected_mode,
            controls=control,
            beam_width=max(4, min(96, control.lookahead * 3)),
        )
        ordered = [analyzed[0], *optimized]
        ordered_media = [media_by_id[entry["queue_item_id"]] for entry in ordered]
        playlists = self.mass.music.playlists
        await playlists._handle_remove_playlist_tracks(
            playlist_item_id, tuple(range(len(ordered_media))), None
        )
        await playlists._handle_add_playlist_tracks(
            playlist_item_id, [media.uri for media in ordered_media], None
        )
        levels = [
            {"name": entry.get("name"), "energy": (entry.get("analysis") or {}).get("energy")}
            for entry in ordered
        ]
        return {
            "playlist_id": playlist_item_id,
            "saved": True,
            "track_count": len(ordered_media),
            "order": [entry.get("name") for entry in ordered],
            "levels": levels,
            "transition_notes": _transition_report(
                [dict(entry, **(entry.get("analysis") or {})) for entry in ordered], control
            ),
        }

    async def _status_total_indexed(self) -> int:
        """
        Count quick-pull index rows, exactly the way pull_candidates does.

        The FE's honest-progress line (0.0.28 lane E) reads this off
        smart_dj/status. An empty index is simply 0; an unreadable one
        degrades to 0 with a debug line - the status command must never
        throw over bookkeeping.
        """
        total_indexed = 0
        index = getattr(self, "_quick_index", None)
        if index is not None:
            try:
                await asyncio.to_thread(index.open)
                total_indexed = await asyncio.to_thread(index.count)
            except (OSError, ValueError) as err:
                self.logger.debug("Smart DJ status: index count unavailable: %s", err)
                total_indexed = 0
        return total_indexed

    async def _status_library_tracks(self) -> int | None:
        """
        Total library track count, computed the way the sweep's total is.

        Same ``library_items`` read + resolvable-item filter as the sweep's
        ``total`` (same limit ceiling, same item_id+provider guard). A
        library provider that is mid-sync (or otherwise failing) degrades
        the field to None instead of failing the status command.
        """
        try:
            library_items = await self.mass.music.tracks.library_items(
                favorite=None,
                search=None,
                limit=LIBRARY_SCAN_MAX_TRACKS,
                offset=0,
                order_by="sort_name",
                summary=False,
            )
            return sum(
                1
                for item in library_items
                if getattr(item, "item_id", None) and getattr(item, "provider", None)
            )
        except Exception as err:  # any mid-sync failure degrades, never throws
            self.logger.debug("Smart DJ status: library count unavailable: %s", err)
            return None

    async def _resolve_track_uris(self, track_uris: list[str]) -> tuple[list[Any], list[str]]:
        """
        Resolve a track-uri list to full media items (0.0.27 mix_tracks).

        Returns (resolved tracks, unresolved URIs). A URI that cannot be
        resolved, or resolves to something other than a Track, is reported
        back instead of failing the whole mix.
        """
        resolved: list[Any] = []
        unresolved: list[str] = []
        for uri in track_uris:
            try:
                item = await self.mass.music.get_item_by_uri(uri)
            except Exception:  # one bad URI must not kill the whole mix
                self.logger.debug("Smart DJ mix_tracks: unresolved uri %s", uri)
                unresolved.append(uri)
                continue
            if getattr(item, "media_type", None) == MediaType.TRACK and getattr(
                item, "item_id", None
            ):
                resolved.append(item)
            else:
                self.logger.debug("Smart DJ mix_tracks: uri is not a track: %s", uri)
                unresolved.append(uri)
        return resolved, unresolved

    async def _mix_resolved(  # noqa: PLR0915 - linear pipeline by design
        self,
        queue_id: str,
        resolved: list[Any],
        control: DJControls,
        mode: str,
        bpm_tolerance: float,
        analysis_provider: str,
        pool_source: str,
        response_base: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Shared mix pipeline for mix_playlist + mix_tracks (0.0.27).

        Analyzes the fully-resolved track list, plans the Smart DJ order,
        applies it to the queue, and pins AutoMix. ``response_base`` carries
        each command's own response keys (playlist ids vs track-uri counts);
        they merge into EVERY response this helper returns, including the
        fail-soft blocked shape, so each command's response contract is
        unchanged. Nothing below mutates the queue before the plan succeeds.
        """
        preset = MODES.get(mode, MODES["ai_dj"])
        selected_mode = DJMode(
            preset.name,
            _bpm_tolerance(bpm_tolerance),
            preset.energy_direction,
            preset.variety,
            preset.weights,
        )
        analyzed, media_by_id = await _playlist_analysis_rows(
            self, resolved, analysis_provider, pool_source
        )
        automix_info: dict[str, Any] | None = None
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
            # RESUME REBUILD: the queue is playing THIS selection mid-stream.
            # Anchor the plan at the track that is playing right now (it keeps
            # playing, untouched) and reorder only the not-yet-played tail --
            # instead of replacing the queue and restarting from the top.
            anchor = next(entry for entry in analyzed if str(entry["item_id"]) == current_media_id)
            played_ids = {
                str(getattr(item.media_item, "item_id", "") or "")
                for item in items[: current_index + 1]
            }
            remaining = [entry for entry in analyzed if str(entry["item_id"]) not in played_ids]
            try:
                optimized = beam_optimize(
                    remaining,
                    _merge_track(anchor),
                    selected_mode,
                    controls=control,
                    beam_width=max(4, min(96, control.lookahead * 3)),
                )
            except HardRequirementsImpossible as err:
                # Plan-phase fail-soft (0.0.27 Item 2): SmartDJError carries only
                # strings, so the structured panel payload rides the normal
                # response shape instead. Nothing below has mutated the queue
                # yet - beam_optimize is pure - so returning here is safe.
                return {
                    "queue_id": queue_id,
                    **response_base,
                    "resumed": False,
                    "current_item_id": current_media_id,
                    "count": 0,
                    "tracks": [],
                    "automix": None,
                    "hard_requirements": {"blocked": True, "rules": err.details},
                }
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
                        for mix_id, secs in _automix_durations(
                            optimized, control, previous=anchor
                        ).items()
                        if mix_id in id_map
                    }
                )
                self.mass.streams.audio.set_automix_blend_prefs(
                    {
                        id_map[mix_id]: pref
                        for mix_id, pref in _automix_blend_prefs(
                            optimized, control, previous=anchor
                        ).items()
                        if mix_id in id_map
                    }
                )
                self.mass.streams.audio.set_automix_stems_vetoes(
                    {
                        id_map[mix_id]: veto
                        for mix_id, veto in _automix_stems_vetoes(
                            optimized, control, previous=anchor
                        ).items()
                        if mix_id in id_map
                    }
                )
                self.mass.streams.audio.set_automix_stems(control.stems_enabled)
                self.mass.create_task(self.preroll_queue_analysis(queue_id, 10))
                automix_info = {
                    "enabled": True,
                    "mode": "smart" if self.mass.streams.smart_fades_available else "standard",
                }
            return {
                "queue_id": queue_id,
                **response_base,
                "resumed": True,
                "current_item_id": current_media_id,
                "count": len(optimized),
                "tracks": optimized,
                "automix": automix_info,
                # per-transition derived bars + reasons (0.0.28 lane C): the
                # panel surface reads these from the mix response
                "transition_notes": _transition_report([anchor, *optimized], control),
                "hard_requirements": {"blocked": False, "rules": []},
            }

        # FRESH MIX: nothing playing (or the playing track is not from this
        # selection) -- replace the queue with the fully optimized track list.
        try:
            optimized = beam_optimize(
                analyzed,
                None,
                selected_mode,
                controls=control,
                beam_width=max(4, min(96, control.lookahead * 3)),
            )
        except HardRequirementsImpossible as err:
            # Plan-phase fail-soft (0.0.27 Item 2): same reasoning as the
            # resume-rebuild site above - the panel payload rides the normal
            # response shape because SmartDJError carries only strings, and
            # the queue is still untouched.
            return {
                "queue_id": queue_id,
                **response_base,
                "resumed": False,
                "count": 0,
                "tracks": [],
                "automix": None,
                "hard_requirements": {"blocked": True, "rules": err.details},
            }
        if not optimized:
            raise RuntimeError("Hard requirements are impossible with the current playlist")
        ordered_media = [media_by_id[track["queue_item_id"]] for track in optimized]
        self.logger.info(
            "Smart DJ: mixing %d resolved tracks on queue %s",
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
            self.mass.streams.audio.set_automix_stems_vetoes(
                {
                    fresh_ids[mix_id]: veto
                    for mix_id, veto in _automix_stems_vetoes(analyzed, control).items()
                    if mix_id in fresh_ids
                }
            )
            self.mass.streams.audio.set_automix_stems(control.stems_enabled)
            # silent background preroll: analyze upcoming tracks so blends
            # improve as the queue plays (no audible output)
            self.mass.create_task(self.preroll_queue_analysis(queue_id, 10))
            automix_info = {
                "enabled": True,
                "mode": "smart" if self.mass.streams.smart_fades_available else "standard",
            }
        return {
            "queue_id": queue_id,
            **response_base,
            "resumed": False,
            "count": len(ordered_media),
            "tracks": optimized,
            "automix": automix_info,
            # per-transition derived bars + reasons (0.0.28 lane C): the
            # panel surface reads these from the mix response
            "transition_notes": _transition_report(optimized, control),
            "hard_requirements": {"blocked": False, "rules": []},
        }

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
        # see has_real_analysis: a tombstone must not satisfy the preroll gate
        if has_real_analysis(existing):
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
                if has_real_analysis(stored):
                    # same derived-index upsert as _analyze_media_silent's
                    # success point: preroll-stored analysis must reach the
                    # quick-pull index too, or pull_candidates would miss
                    # tracks the blend engine already knows
                    await self._index_upsert(media_item_id, media_provider, stored, media)
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
                state = str(value.get("state", fallback))
                weight = _signal_weight(value.get("weight", 1.0))
            else:
                state = str(value)
                weight = 1.0
            if state not in {"hard", "soft", "disabled"}:
                # A typo like "hurd" must fail loudly, not silently disable the
                # signal the user asked for.
                raise SmartDJError(
                    "invalid_control",
                    f"{name}.state={state!r} must be hard, soft, or disabled",
                )
            return SignalControl(state=state, weight=weight)

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
        if transition_style not in {
            "auto",
            "full_blend",
            "energy",
            "quick_cut",
            "radio",
            "echo_out",
            "filter_out",
        }:
            raise RuntimeError(
                "transition_style must be one of auto, full_blend, energy, quick_cut, radio, echo_out, filter_out"
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
            stems_enabled=bool(raw.get("stems_enabled", False)),
            stems_veto_override=bool(raw.get("stems_veto_override", False)),
            energy_arc=bool(raw.get("energy_arc", False)),
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
                        raise SmartDJError(  # noqa: TRY301 - the raise IS the error channel
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
                        raise RuntimeError(  # noqa: TRY301 - the raise IS the error channel
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
                # has_real_analysis, not `is not None`: an all-null
                # tombstone must not surface as a phantom "analyzed" dict
                # the planner treats as real
                if analysis is not None and has_real_analysis(analysis):
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
            image = metadata_obj.images[0] if metadata_obj and metadata_obj.images else None
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
                    # first artwork image, serialized for the frontend's
                    # getMediaItemImageUrl() (album art behind the row)
                    "image": (
                        {
                            "path": image.path,
                            "proxy_id": image.proxy_id,
                            "remotely_accessible": image.remotely_accessible,
                        }
                        if image
                        else None
                    ),
                }
            )
        return result

    def _any_player_playing(self) -> bool:
        """Return True when ANY player on this MA instance is actively playing."""
        return any(
            player.state.playback_state == PlaybackState.PLAYING for player in self.mass.players
        )

    def _background_concurrency(self) -> int:
        """Read the streams background-scan concurrency setting, clamped to 1-8."""
        try:
            raw = int(
                self.mass.config.get_raw_core_config_value(
                    "streams",
                    CONF_BACKGROUND_SCAN_CONCURRENCY,
                    DEFAULT_BACKGROUND_SCAN_CONCURRENCY,
                )
                or DEFAULT_BACKGROUND_SCAN_CONCURRENCY
            )
        except ValueError, TypeError:
            raw = DEFAULT_BACKGROUND_SCAN_CONCURRENCY
        return max(1, min(raw, 8))

    @staticmethod
    def _analysis_targets_for_track(track: Any) -> list[tuple[str, str]]:
        """
        Resolve a track to analyzable (provider item_id, provider instance_id) pairs.

        Library items report provider="library" (the library pseudo-provider)
        with the real source providers in provider_mappings; audio analysis is
        keyed and read through those provider-native ids, never the library id.
        Only mappings the provider currently reports as available are returned;
        non-library items fall back to their own provider fields.
        """
        out: list[tuple[str, str]] = []
        for mapping in getattr(track, "provider_mappings", None) or []:
            if not getattr(mapping, "available", False):
                continue
            item_id = getattr(mapping, "item_id", None)
            instance_id = getattr(mapping, "provider_instance", None)
            if item_id and instance_id:
                out.append((str(item_id), str(instance_id)))
        if not out:
            prov = getattr(track, "provider", None)
            item_id = getattr(track, "item_id", None)
            if prov and prov != "library" and item_id:
                out.append((str(item_id), str(prov)))
        return out

    async def _sweep_attempt(self, track: Any) -> bool:
        """
        Run one sweep track's analysis attempts under the hard timeout.

        True when any provider mapping stored a new analysis. Raises
        TimeoutError when a single attempt exceeds ANALYSIS_ATTEMPT_TIMEOUT;
        wait_for's cancellation is torn down by _analyze_media_silent's
        finally (buffer.clear evicts the session).
        """
        for a_item_id, a_instance in self._analysis_targets_for_track(track):
            if await asyncio.wait_for(
                self._analyze_media_silent(a_item_id, a_instance, track=track),
                timeout=ANALYSIS_ATTEMPT_TIMEOUT,
            ):
                return True
        return False

    async def _library_scan_worker(
        self, targets: list[Any], state: dict[str, Any], pause_while_playing: bool
    ) -> None:
        """
        Run the sweep behind the safeguards.

        Every dispatch acquires the concurrency semaphore, then defers while
        playback is active, then analyzes one track. Cancel and the failure
        circuit breaker are checked between semaphore acquisitions so nothing
        half-done is ever left behind (``_analyze_media_silent`` always tears
        its buffer down in a finally).
        """
        sem = asyncio.Semaphore(self._background_concurrency())
        failures = 0

        async def _one(track: Any) -> None:
            nonlocal failures
            async with sem:
                if failures >= LIBRARY_SCAN_MAX_FAILURES:
                    state["skipped"] += 1
                    return
                while pause_while_playing and not self._library_scan_cancel.is_set():
                    if self._any_player_playing():
                        state["deferring"] = True
                        await asyncio.sleep(LIBRARY_SCAN_POLL_SECONDS)
                    else:
                        state["deferring"] = False
                        break
                if self._library_scan_cancel.is_set():
                    state["deferring"] = False
                    state["skipped"] += 1
                    return
                raw_id = getattr(track, "item_id", None)
                track_id = str(raw_id) if raw_id else "unknown"
                track_name = _sweep_track_name(track)
                try:
                    done = await self._sweep_attempt(track)
                except TimeoutError:
                    # hard timeout: the producer was cancelled (wait_for) and
                    # _analyze_media_silent's finally evicted the session
                    failures += 1
                    state["failed"] += 1
                    reason = f"timeout after {ANALYSIS_ATTEMPT_TIMEOUT}s"
                    self.logger.warning(
                        "Smart DJ sweep: analysis of %s (%s) %s - producer cancelled",
                        track_name,
                        track_id,
                        reason,
                    )
                    _record_sweep_failure(state, track_id, track_name, reason)
                    return
                except Exception as err:
                    failures += 1
                    state["failed"] += 1
                    _record_sweep_failure(state, track_id, track_name, str(err))
                    return
                if done:
                    failures = 0
                    state["analyzed"] += 1
                else:
                    state["skipped"] += 1

        try:
            await asyncio.gather(*(_one(track) for track in targets))
            if failures >= LIBRARY_SCAN_MAX_FAILURES:
                state["errors"].append(f"sweep aborted after {failures} consecutive failures")
        finally:
            state["running"] = False
            state["deferring"] = False
            state["finished_at"] = time.time()
            self._library_scan_task = None
            await self._save_cache()

    def _preanalysis_enabled(self) -> bool:
        """
        Read the pre-analysis toggle from the provider settings (default ON).

        The explicit ``True`` default matters: get_value returns None for a
        key that was never stored (e.g. a provider instance configured before
        this setting existed), and ``bool(None)`` would silently turn the
        feature OFF despite its declared default.
        """
        return bool(self.config.get_value(CONF_PREANALYSIS_ENABLED, True))

    async def _on_media_added(self, event: MassEvent) -> None:
        """
        React to a MEDIA_ITEM_ADDED event: queue new tracks for pre-analysis.

        Everything that is not a track (albums, playlists, genres, ...) is
        ignored; the toggle is read live so the setting applies without a
        provider reload. Bulk provider syncs never reach this (per-item
        events are suppressed during a sync).
        """
        item = getattr(event, "data", None)
        if getattr(item, "media_type", None) != MediaType.TRACK:
            return
        if not self._preanalysis_enabled():
            return
        self._preanalysis_enqueue(item)

    def _preanalysis_enqueue(self, track: Any) -> None:
        """
        Queue one newly added track for background analysis.

        Bounded by PREANALYSIS_QUEUE_MAX: when full, the OLDEST pending track
        is dropped with a WARNING so a burst of additions cannot grow the
        queue without limit. Duplicate uris are not queued twice.
        """
        uri = getattr(track, "uri", None)
        if uri and uri in self._preanalysis_pending_uris:
            return
        while len(self._preanalysis_queue) >= PREANALYSIS_QUEUE_MAX:
            dropped = self._preanalysis_queue.popleft()
            self._preanalysis_pending_uris.discard(getattr(dropped, "uri", None))
            self.logger.warning(
                "Smart DJ pre-analysis: pending queue full (%d), dropping oldest "
                "track %s to keep the queue bounded",
                PREANALYSIS_QUEUE_MAX,
                _sweep_track_name(dropped),
            )
        if uri:
            self._preanalysis_pending_uris.add(uri)
        self._preanalysis_queue.append(track)
        if self._preanalysis_task is None or self._preanalysis_task.done():
            self._preanalysis_task = self.mass.create_task(self._preanalysis_worker())

    async def _preanalysis_worker(self) -> None:
        """
        Drain the bounded pre-analysis queue with the sweep's safeguards.

        Same machinery as the sweep (0.0.28 lane B): each track goes through
        ``_sweep_attempt`` under the same hard per-track timeout, behind a
        semaphore sized by the same background-concurrency setting, deferring
        while any player streams, with the same consecutive-failure circuit
        breaker (reset on success, abort after LIBRARY_SCAN_MAX_FAILURES).
        Success flows through ``_analyze_media_silent``'s existing
        ``_index_upsert`` success point, so a track analyzed on arrival lands
        in the quick-pull index exactly like a sweep-analyzed track.
        """
        sem = asyncio.Semaphore(self._background_concurrency())
        failures = 0
        try:
            while self._preanalysis_queue:
                # breaker checked before dispatch (same as the sweep): nothing
                # half-done is left behind, a broken provider stops the drain
                if failures >= LIBRARY_SCAN_MAX_FAILURES:
                    self.logger.warning(
                        "Smart DJ pre-analysis: aborted after %d consecutive "
                        "failures, dropping %d pending track(s)",
                        failures,
                        len(self._preanalysis_queue),
                    )
                    self._preanalysis_queue.clear()
                    self._preanalysis_pending_uris.clear()
                    break
                track = self._preanalysis_queue.popleft()
                self._preanalysis_pending_uris.discard(getattr(track, "uri", None))
                async with sem:
                    while self._any_player_playing():
                        self._preanalysis_deferring = True
                        await asyncio.sleep(LIBRARY_SCAN_POLL_SECONDS)
                    self._preanalysis_deferring = False
                    track_id = str(getattr(track, "item_id", None) or "unknown")
                    track_name = _sweep_track_name(track)
                    try:
                        done = await self._sweep_attempt(track)
                    except TimeoutError:
                        failures += 1
                        reason = f"timeout after {ANALYSIS_ATTEMPT_TIMEOUT}s"
                        self.logger.warning(
                            "Smart DJ pre-analysis: analysis of %s (%s) %s - producer cancelled",
                            track_name,
                            track_id,
                            reason,
                        )
                        continue
                    except Exception as err:
                        failures += 1
                        self.logger.warning(
                            "Smart DJ pre-analysis: analysis of %s (%s) failed: %s",
                            track_name,
                            track_id,
                            err,
                        )
                        continue
                    if done:
                        failures = 0
            if failures >= LIBRARY_SCAN_MAX_FAILURES:
                # the drain ended exactly on the breaker (queue exhausted on a
                # failure) - say so, same as the sweep's epilogue
                self.logger.warning(
                    "Smart DJ pre-analysis: stopped after %d consecutive failures",
                    failures,
                )
                self._preanalysis_pending_uris.clear()
        finally:
            self._preanalysis_deferring = False
            self._preanalysis_task = None

    async def _analyze_media_silent(
        self, item_id: str, provider_id: str, track: Any = None
    ) -> bool:
        """
        Silently analyze one media item through the local ML providers.

        Returns True when analysis was newly stored; False when it already
        existed or the item cannot be analyzed (realtime source, provider gone).
        ``track`` is optional MA metadata context (mood/release_date) for the
        derived quick-pull index upsert; callers that hold the track object
        pass it so era/mood derivations do not need another library read.
        """
        existing = await self.mass.streams.audio_analysis.get_audio_analysis(
            item_id=item_id,
            provider_instance_id_or_domain=provider_id,
        )
        # see has_real_analysis: tombstones must not block the sweep dispatch
        if has_real_analysis(existing):
            return False
        music_prov = self.mass.get_provider(provider_id, provider_type=MusicProvider)
        if music_prov is None or not music_prov.available:
            return False
        streamdetails = await music_prov.get_stream_details(item_id, MediaType.TRACK)
        if streamdetails.is_realtime:
            return False
        pcm_format = decoded_pcm_format(streamdetails)
        buffer = AudioBuffer(pcm_format, BufferSize.BALANCED, BufferMode.SEEKABLE)
        buffer.fill(
            self.mass.streams.audio.get_media_stream(streamdetails, pcm_format),
            source_name="smart_dj_background",
        )
        try:
            await self.mass.streams.audio_analysis.start_analysis(buffer, streamdetails)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + PREROLL_ANALYSIS_TIMEOUT
            while loop.time() < deadline and not buffer.cancelled and not buffer.eof:
                await asyncio.sleep(PREROLL_POLL_SECONDS)
                stored = await self.mass.streams.audio_analysis.get_audio_analysis(
                    item_id=item_id,
                    provider_instance_id_or_domain=provider_id,
                )
                if has_real_analysis(stored):
                    # success point (spec 3.2): fold the freshly stored
                    # analysis into the derived quick-pull index while the
                    # AudioAnalysisData is in hand; fail-soft, never a
                    # successful analysis becomes a sweep failure
                    await self._index_upsert(item_id, provider_id, stored, track)
                    return True
            return False
        finally:
            # tears down the producer and (via the buffer's cancel callback)
            # evicts any still-active analysis session for this item
            await buffer.clear()

    async def _index_upsert(
        self, item_id: str, provider_id: str, analysis: Any, track: Any = None
    ) -> None:
        """
        Derive and upsert one quick-pull index row from a stored analysis.

        The index may never break the analysis that feeds it: any failure
        here is logged and dropped - the next backfill re-derives the row.

        An all-null analysis (an acoustid no-match tombstone, see
        has_real_analysis) is refused here: the row would carry a current
        analyzer_version with every scalar NULL, which is indistinguishable
        from a real derived row and would keep total_indexed dishonest. The
        gate lives in this one place so every call site - the sweep success
        poll, the preroll poll and the backfill - is covered.
        """
        if not has_real_analysis(analysis):
            self.logger.debug(
                "Quick-pull index upsert skipped for %s: analysis carries no "
                "real measurement (acoustid tombstone or empty row)",
                item_id,
            )
            return
        try:
            metadata = getattr(track, "metadata", None)
            row = derive_row(item_id, provider_id, analysis, metadata)
            await asyncio.to_thread(self._quick_index.upsert, row)
        except Exception as err:  # fail-soft by contract; the backfill re-derives
            self.logger.debug("Quick-pull index upsert failed for %s: %s", item_id, err)

    async def _quick_index_backfill(self) -> int:
        """
        Backfill the quick-pull index from the analysis store (spec 3.2).

        One pass over ``iter_merged_audio_analysis_rows`` (no music provider
        loaded, no analyzer quota - the rows are already on disk), upserting
        only tracks the index has not yet derived. Runs in the background on
        provider load; the sweep's own upserts keep the index fresh after it.
        """
        index = self._quick_index
        await asyncio.to_thread(index.open)
        try:
            done = await asyncio.to_thread(index.all_keys, INDEX_VERSION)
        except Exception as err:  # an unreadable index re-derives below
            self.logger.warning("Quick-pull index unreadable, rebuilding: %s", err)
            done = set()
        derived = 0
        try:
            rows_iter = self.mass.streams.audio_analysis.iter_merged_audio_analysis_rows(
                SONIC_ANALYSIS_DOMAIN, MediaType.TRACK
            )
            async for item_id, provider, analysis in rows_iter:
                if (str(item_id), str(provider)) in done:
                    continue
                await self._index_upsert(str(item_id), str(provider), analysis, None)
                derived += 1
        except Exception as err:  # incomplete beats absent, not failed
            self.logger.warning("Quick-pull index backfill incomplete: %s", err)
        if derived:
            self.logger.info(
                "Quick-pull index backfill derived %d row(s) from audio analysis",
                derived,
            )
        return derived

    async def _pool_analyses(
        self,
        rows: list[dict[str, Any]],
        analysis_provider: str,
        pool_source: str = "auto",
    ) -> list[Any]:
        """
        Resolve per-track analyses for candidate-pool building, index first.

        The planner pool-pulling fast path (0.0.27 Item 3 / spec 3.3): tracks
        with a fresh derived index row get their planner-facing scalars from
        the quick-pull index (one indexed query) instead of a payload-blob
        deserialize per track from audio_analysis.db. Tracks missing from the
        index - or carrying a stale derivation version - fall back to the
        existing ``_analysis`` path, and ``pool_source="full"`` disables the
        fast path entirely. Every row keeps its position; analysis failures
        come back as exceptions for the caller to log, matching the old
        ``gather(return_exceptions=True)`` contract.
        """
        results: list[Any] = [None] * len(rows)
        index_rows: dict[tuple[str, str], dict[str, Any]] = {}
        index = getattr(self, "_quick_index", None)
        if index is not None and pool_source != "full":
            keys = [(str(row["item_id"]), str(row["provider"])) for row in rows]
            try:
                index_rows = await asyncio.to_thread(index.lookup, keys)
            except Exception as err:  # fallback beats failure
                self.logger.debug("Quick-pull index lookup failed: %s", err)
        pending: list[tuple[int, asyncio.Task[Any]]] = []
        for idx, row in enumerate(rows):
            hit = index_rows.get((str(row["item_id"]), str(row["provider"])))
            if hit is not None:
                results[idx] = analysis_from_index_row(hit)
                continue
            pending.append(
                (
                    idx,
                    asyncio.create_task(
                        self._analysis(
                            row["item_id"],
                            row["provider"],
                            {k: row[k] for k in ("genre", "genres", "explicit") if k in row},
                            analysis_provider,
                        )
                    ),
                )
            )
        for idx, task in pending:
            try:
                results[idx] = await task
            except Exception as err:  # caller logs, row stays None
                self.logger.debug("Smart DJ analysis failed for %s: %s", rows[idx]["item_id"], err)
                results[idx] = err
        return results
