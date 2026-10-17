"""Smart DJ scoring, hard constraints, and bounded playlist optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalControl:
    """Per-signal policy: hard enforcement, soft preference, or disabled."""

    state: str = "soft"  # hard | soft | disabled
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DJWeights:
    """Relative weights of the scoring signals for a mode."""

    bpm: float = 0.29
    key: float = 0.24
    energy: float = 0.14
    danceability: float = 0.10
    loudness: float = 0.05
    genre: float = 0.05
    artist_spacing: float = 0.04
    momentum: float = 0.04
    clap: float = 0.05


# Segue-quality scoring: how well two tracks' quiet EDGES overlap, from the same
# RMS math the transition planner runs at playback time. Not a user control - it
# is an intrinsic quality term that abstains silently when either track lacks
# RMS analysis, and it scales with the transition aggressiveness like momentum.
SEGUE_QUALITY_WEIGHT: float = 0.12
_SEGUE_FULL_SCORE_SECONDS: float = 10.0


@dataclass(frozen=True, slots=True)
class DJMode:
    """A named DJ preset: tolerances, preferred energy direction, and weights."""

    name: str
    bpm_tolerance: float
    energy_direction: float
    variety: float
    weights: DJWeights


@dataclass(frozen=True, slots=True)
class DJControls:
    """User-facing control surface: per-signal policies plus hard constraints."""

    bpm: SignalControl = SignalControl()
    key: SignalControl = SignalControl()
    energy: SignalControl = SignalControl()
    danceability: SignalControl = SignalControl()
    loudness: SignalControl = SignalControl()
    genre: SignalControl = SignalControl()
    artist_spacing: SignalControl = SignalControl()
    momentum: SignalControl = SignalControl()
    clap: SignalControl = SignalControl()
    bpm_min: float | None = None
    bpm_max: float | None = None
    max_bpm_jump: float | None = None
    key_relation: str = "compatible"
    max_artist_repeat: int = 1  # maximum consecutive tracks by the same artist
    instrumental: str = "any"  # any | prefer | required
    explicit: str = "allow"  # allow | exclude
    transition_bars: int = 8
    # Per-track transition length overrides, keyed by queue_item_id (the
    # transition INTO that track). Values: 4/8/16/32 bars, or 0 = hard cut.
    transition_bars_overrides: dict[str, int] = field(default_factory=dict)
    automix_enabled: bool = False
    smart_reorder_enabled: bool = True
    # Vocal-duck toggle for the crossfade overlap (smart_fades/stems.py): when
    # True, pinned transitions duck the outgoing vocal instead of playing the
    # double-vocal handoff. Soft: degrades to the plain blend when the
    # separation runtime/model is unavailable.
    stems_enabled: bool = False
    lookahead: int = 8
    # Blend style preference passed to the smart-fades planner: auto (engine
    # decides), full_blend (force the 16-bar blend when the grid allows),
    # energy (medium blend), quick_cut (always the short downbeat fade).
    transition_style: str = "auto"
    # Stretch gate: max BPM-difference percent the planner will time-stretch
    # across. strict=6, normal=8 (engine default), aggressive=12.
    beat_strictness: str = "normal"
    transition_aggressiveness: float = 0.5
    required_ids: frozenset[str] = frozenset()
    excluded_ids: frozenset[str] = frozenset()
    fixed_ids: frozenset[str] = frozenset()
    end_track_id: str | None = None


MODES = {
    "ai_dj": DJMode("ai_dj", 0.08, 0.15, 0.45, DJWeights()),
    "party": DJMode(
        "party",
        0.12,
        0.05,
        0.75,
        DJWeights(
            bpm=0.26,
            key=0.18,
            energy=0.16,
            danceability=0.18,
            loudness=0.06,
            genre=0.05,
            artist_spacing=0.04,
            momentum=0.04,
            clap=0.03,
        ),
    ),
    "chill": DJMode(
        "chill",
        0.10,
        -0.15,
        0.65,
        DJWeights(
            bpm=0.22,
            key=0.18,
            energy=0.23,
            danceability=0.10,
            loudness=0.06,
            genre=0.08,
            artist_spacing=0.05,
            momentum=0.05,
            clap=0.03,
        ),
    ),
    "workout": DJMode(
        "workout",
        0.06,
        0.12,
        0.35,
        DJWeights(
            bpm=0.31,
            key=0.18,
            energy=0.19,
            danceability=0.16,
            loudness=0.04,
            genre=0.03,
            artist_spacing=0.03,
            momentum=0.03,
            clap=0.03,
        ),
    ),
    "custom": DJMode("custom", 0.08, 0.0, 0.50, DJWeights()),
}


def _norm_delta(a: Any, b: Any, scale: float) -> float:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return 0.5
    return max(0.0, 1.0 - abs(float(a) - float(b)) / max(scale, 0.001))


def _coerce_embedding(value: Any) -> list[float] | None:
    """Normalise an embedding that may arrive as a list, tuple or JSON string."""
    if isinstance(value, (list, tuple)):
        return list(value) if value else None
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, list) and parsed:
            return parsed
    return None


def clap_similarity(a: Any, b: Any) -> float:
    """Cosine similarity of two CLAP embeddings, normalised to 0.0-1.0."""
    emb_a = _coerce_embedding(a)
    emb_b = _coerce_embedding(b)
    if emb_a is None or emb_b is None or len(emb_a) != len(emb_b):
        return 0.5
    vec_a = np.asarray(emb_a, dtype=np.float32)
    vec_b = np.asarray(emb_b, dtype=np.float32)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.5
    cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return (cosine + 1.0) / 2.0


def camelot_affinity(a: str | None, b: str | None) -> float:
    """Harmonic compatibility of two Camelot notations (0.0-1.0)."""
    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    try:
        na, nb = int(str(a)[:-1]), int(str(b)[:-1])
        ma, mb = str(a)[-1].upper(), str(b)[-1].upper()
    except (ValueError, TypeError):
        return 0.0
    if ma == mb and ((na - nb) % 12 in (1, 11)):
        return 0.85
    if na == nb and ma != mb:
        return 0.72
    if ((na - nb) % 12 in (1, 11)) and ma != mb:
        return 0.55
    return 0.0


def _artist_run_length(
    path: list[dict[str, Any]],
    artist: str | None,
    *,
    anchor_artist: str | None = None,
) -> int:
    """
    Count consecutive same-artist tracks at the end of path, including the anchor.

    The anchor (now-playing / previous fixed track) is not in ``path`` but still
    counts toward the consecutive-run limit. When the entire path is the same
    artist as the anchor, the run includes the anchor.
    """
    if not artist:
        return 0
    count = 0
    for item in reversed(path):
        if item.get("artist") != artist:
            break
        count += 1
    # If the whole path matches, or the path is empty, fold in the anchor.
    if count == len(path) and anchor_artist == artist:
        count += 1
    return count


# Stretch-gate ceiling per beat_strictness setting: the maximum BPM-difference
# fraction the hard BPM gate will accept. strict=6%, normal=8% (default),
# aggressive=12%, matching the documented DJControls.beat_strictness contract.
BEAT_STRETCH_LIMITS = {"strict": 0.06, "normal": 0.08, "aggressive": 0.12}


def _hard_fail(  # noqa: PLR0915,C901,PLR0912 - every hard rule reads clearest inline
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    controls: DJControls,
    *,
    artist_run_length: int = 0,
) -> list[str]:
    violations: list[str] = []
    if not candidate:
        return ["missing analysis"]

    candidate_id = candidate.get("queue_item_id")
    if candidate_id in controls.excluded_ids:
        violations.append("excluded track")

    bpm = candidate.get("bpm")
    if controls.bpm_min is not None and not isinstance(bpm, (int, float)):
        violations.append("BPM unavailable for minimum constraint")
    elif isinstance(bpm, (int, float)) and controls.bpm_min is not None and bpm < controls.bpm_min:
        violations.append("below minimum BPM")
    if controls.bpm_max is not None and not isinstance(bpm, (int, float)):
        violations.append("BPM unavailable for maximum constraint")
    elif isinstance(bpm, (int, float)) and controls.bpm_max is not None and bpm > controls.bpm_max:
        violations.append("above maximum BPM")

    if current and controls.max_bpm_jump is not None:
        a = current.get("bpm")
        if not isinstance(a, (int, float)) or not isinstance(bpm, (int, float)):
            violations.append("BPM unavailable for maximum jump constraint")
        elif abs(float(bpm) - float(a)) > controls.max_bpm_jump:
            violations.append("maximum BPM jump exceeded")

    # "soft" deliberately skips the hard gate: an incompatible step becomes a
    # scoring penalty (see _signal_values), so a scattered-key playlist always
    # yields an order instead of failing as impossible
    if controls.key_relation not in ("any", "soft") and current:
        a, b = current.get("camelot"), candidate.get("camelot")
        if not a or not b:
            violations.append("key unavailable for key constraint")
        else:
            affinity = camelot_affinity(a, b)
            if controls.key_relation == "same" and affinity < 1.0:
                violations.append("key must match")
            elif controls.key_relation == "compatible" and affinity <= 0.0:
                violations.append("incompatible key")

    if controls.instrumental == "required" and candidate.get("instrumental") is not True:
        violations.append("instrumental status unavailable or not instrumental")

    if controls.explicit == "exclude" and candidate.get("explicit") is not False:
        violations.append("explicit status unavailable or track is explicit")

    if controls.bpm.state == "hard" and current:
        a = current.get("bpm")
        if not isinstance(a, (int, float)) or not isinstance(bpm, (int, float)):
            violations.append("hard BPM compatibility unavailable")
        else:
            # Stretch gate scaled by beat_strictness (strict=6%, normal=8%,
            # aggressive=12% - the documented DJControls contract) and tightened
            # to half the window when either side carries vocals: vocal
            # material audibly tears around +-3% pitch shift.
            either_vocal = candidate.get("instrumental") is not True or (
                current.get("instrumental") is not True
            )
            limit = BEAT_STRETCH_LIMITS.get(controls.beat_strictness, 0.08)
            if either_vocal:
                limit *= 0.5
            if abs(float(bpm) - float(a)) > float(a) * limit:
                violations.append("hard BPM compatibility")

    if controls.key.state == "hard" and current:
        a, b = current.get("camelot"), candidate.get("camelot")
        if not a or not b or camelot_affinity(a, b) <= 0.0:
            violations.append("hard key compatibility unavailable or failed")

    if controls.energy.state == "hard" and current:
        a, b = current.get("energy"), candidate.get("energy")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            violations.append("hard energy compatibility unavailable")
        elif abs(float(a) - float(b)) > 0.35:
            violations.append("hard energy compatibility")

    for field_name, label, scale in (
        ("danceability", "danceability", 0.45),
        ("loudness", "loudness", 8.0),
    ):
        control = getattr(controls, field_name)
        if control.state == "hard" and current:
            a, b = current.get(field_name), candidate.get(field_name)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                violations.append(f"hard {label} compatibility unavailable")
            elif abs(float(a) - float(b)) > scale:
                violations.append(f"hard {label} compatibility")

    if controls.genre.state == "hard" and current:
        a, b = current.get("genre"), candidate.get("genre")
        if not a or not b:
            violations.append("hard genre compatibility unavailable")
        elif a != b:
            violations.append("hard genre compatibility")

    if controls.artist_spacing.state == "hard" and artist_run_length >= max(
        1, controls.max_artist_repeat
    ):
        violations.append("hard artist spacing")

    # "soft" spacing is a graded score penalty in _signal_values (the
    # artist_spacing value decays to zero as the run length reaches the cap);
    # "disabled" must neither veto nor penalize. The old unconditional veto
    # made max_artist_repeat=1 (the default) reject EVERY same-artist
    # follow-up regardless of the user's soft/disabled setting.
    return violations


def _signal_values(
    current: dict[str, Any],
    candidate: dict[str, Any],
    mode: DJMode,
    controls: DJControls,
    *,
    artist_run_length: int,
) -> tuple[float, list[str]]:
    w = mode.weights
    values = {
        "bpm": _norm_delta(
            current.get("bpm"),
            candidate.get("bpm"),
            max(1.0, float(current.get("bpm") or 120) * mode.bpm_tolerance),
        ),
        "key": camelot_affinity(current.get("camelot"), candidate.get("camelot")),
        "energy": max(
            0.0,
            1.0
            - abs(
                float(candidate.get("energy") or 0.5)
                - (float(current.get("energy") or 0.5) + mode.energy_direction)
            )
            / 0.35,
        ),
        "danceability": _norm_delta(
            current.get("danceability"), candidate.get("danceability"), 0.45
        ),
        "loudness": _norm_delta(current.get("loudness"), candidate.get("loudness"), 8.0),
        "genre": 1.0
        if current.get("genre")
        and candidate.get("genre")
        and current["genre"] == candidate["genre"]
        else 0.5,
        "artist_spacing": max(0.0, 1.0 - artist_run_length / max(1, controls.max_artist_repeat)),
        "momentum": 1.0
        if float(candidate.get("energy") or 0.5) >= float(current.get("energy") or 0.5)
        else 0.65,
        "clap": clap_similarity(current.get("clap_embedding"), candidate.get("clap_embedding")),
    }
    reasons: list[str] = []
    total = total_weight = 0.0
    for name, value in values.items():
        control = getattr(controls, name)
        if control.state != "soft":
            continue
        if (
            name == "clap"
            and value == 0.5
            and not (current.get("clap_embedding") and candidate.get("clap_embedding"))
        ):
            continue
        effective = max(0.0, control.weight) * getattr(w, name)
        if name == "momentum":
            effective *= 0.5 + controls.transition_aggressiveness
        total += value * effective
        total_weight += effective
        if name == "clap":
            reasons.append("sonic similarity strong" if value >= 0.60 else "sonic similarity weak")
        elif value >= 0.85:
            reasons.append(f"{name.replace('_', ' ')} strong")
    if controls.instrumental == "prefer":
        if candidate.get("instrumental") is True:
            total += 0.10
            total_weight += 0.10
            reasons.append("instrumental preference")
    if controls.key_relation == "soft" and current:
        a, b = current.get("camelot"), candidate.get("camelot")
        if a and b and camelot_affinity(a, b) <= 0.0:
            # soft key: an incompatible step drags the score instead of vetoing
            # the candidate, so it is only chosen when nothing harmonic exists
            total -= 0.35
            total_weight += 0.35
            reasons.append("key softened: incompatible step accepted")
    if current is not None:
        quality = segue_quality(current, candidate)
        if quality is not None:
            effective = SEGUE_QUALITY_WEIGHT * (0.5 + controls.transition_aggressiveness)
            total += quality * effective
            total_weight += effective
            reasons.append(
                "quiet-edge overlap strong" if quality >= 0.7 else "quiet-edge overlap weak"
            )
    if total_weight == 0:
        return 0.5, reasons
    return max(0.0, min(1.0, total / total_weight)), reasons


def _rms_arrays(track: dict[str, Any]) -> tuple[Any, float] | None:
    """Return (rms bins, track duration) for a merged track dict, or None without analysis."""
    rms = track.get("rms_energy")
    duration = track.get("duration")
    if rms is None:
        analysis = track.get("analysis")
        if isinstance(analysis, dict):
            rms = analysis.get("rms_energy")
            duration = duration or analysis.get("duration")
    if not rms:
        return None
    if not duration:
        # no duration on the track dict: the beat grids are media time, so the
        # last beat plus one beat of the track's own tempo is a close estimate
        beats = track.get("beats") or (track.get("analysis") or {}).get("beats")
        bpm = float(track.get("bpm") or 0)
        if not beats or bpm <= 0:
            return None
        duration = float(beats[-1]) + 60.0 / bpm
    return rms, float(duration)


def segue_quality(current: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    """
    Estimate how well the pair's quiet edges overlap, or None without analysis.

    The quiet-tail / quiet-head detection the transition planner runs at
    playback time, evaluated at reorder time from the RMS arrays the provider
    already carries - no audio buffers involved. 1.0 = a >=10s quiet-edge
    overlap, 0.0 = both edges slam.
    """
    import numpy as np  # noqa: PLC0415

    from music_assistant.controllers.streams.smart_fades.helpers import (  # noqa: PLC0415
        SEGUE_ENERGY_FRACTION,
        detect_mix_out_point,
        detect_rise_point,
    )

    out = _rms_arrays(current)
    in_ = _rms_arrays(candidate)
    if out is None or in_ is None:
        return None
    out_rms, out_dur = np.asarray(out[0], dtype=np.float32), out[1]
    in_rms, in_dur = np.asarray(in_[0], dtype=np.float32), in_[1]
    if len(out_rms) < 2 or len(in_rms) < 2:
        return None
    out_bpm = float(current.get("bpm") or 120)
    in_bpm = float(candidate.get("bpm") or 120)
    out_bpb = int(current.get("beats_per_bar") or 4)
    in_bpb = int(candidate.get("beats_per_bar") or 4)
    # the whole tail is the search window: the detection finds where it decays
    point = detect_mix_out_point(
        out_rms, out_dur, out_dur, out_bpm, fraction=SEGUE_ENERGY_FRACTION, beats_per_bar=out_bpb
    )
    quiet_tail = max(0.0, out_dur - point)
    quiet_head = detect_rise_point(
        in_rms, in_dur, in_bpm, SEGUE_ENERGY_FRACTION, beats_per_bar=in_bpb
    )
    overlap = min(quiet_tail + quiet_head, _SEGUE_FULL_SCORE_SECONDS)
    return min(1.0, overlap / _SEGUE_FULL_SCORE_SECONDS)


def score_candidate(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    mode: DJMode,
    controls: DJControls,
    *,
    artist_run_length: int = 0,
) -> tuple[float, list[str], list[str]]:
    """
    Score one candidate against the current track.

    Returns (score, reasons, hard violations); an empty violation list means the
    candidate is allowed under the current hard constraints.
    """
    if not candidate:
        return 0.0, [], ["missing analysis"]
    violations = _hard_fail(current, candidate, controls, artist_run_length=artist_run_length)
    if violations:
        return 0.0, [], violations
    if current is None:
        return 0.5, ["no current-track anchor"], []
    score, reasons = _signal_values(
        current, candidate, mode, controls, artist_run_length=artist_run_length
    )
    return score, reasons, []


def track_score(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    mode: DJMode,
    *,
    bpm_tolerance: float | None = None,
    energy_target: float | None = None,  # noqa: ARG001 - public signature contract
    same_artist: bool = False,
) -> tuple[float, list[str]]:
    """Score a candidate with ad-hoc tolerances (no full DJControls needed)."""
    controls = DJControls(bpm=SignalControl("soft", 1.0), key=SignalControl("soft", 1.0))
    if bpm_tolerance is not None:
        mode = DJMode(mode.name, bpm_tolerance, mode.energy_direction, mode.variety, mode.weights)
    score, reasons, _ = score_candidate(
        current, candidate, mode, controls, artist_run_length=1 if same_artist else 0
    )
    return score, reasons


def _merge_track(item: dict[str, Any]) -> dict[str, Any]:
    analysis = item.get("analysis")
    merged = dict(analysis) if isinstance(analysis, dict) else {}
    for key in ("queue_item_id", "artist", "genre", "explicit", "instrumental"):
        if key in item:
            merged[key] = item[key]
    return merged


def _beam_pass(
    segment: list[dict[str, Any]],
    anchor: dict[str, Any] | None,
    mode: DJMode,
    controls: DJControls,
    beam_width: int,
    anchor_artist: str | None,
) -> list[dict[str, Any]]:
    beams: list[tuple[float, list[dict[str, Any]], list[dict[str, Any]]]] = [
        (0.0, [], list(segment))
    ]
    for _ in range(len(segment)):
        next_beams: list[tuple[float, list[dict[str, Any]], list[dict[str, Any]]]] = []
        for total, path, remaining in beams:
            current = _merge_track(path[-1]) if path else anchor
            for candidate in remaining:
                data = _merge_track(candidate)
                run = _artist_run_length(
                    path,
                    candidate.get("artist"),
                    anchor_artist=anchor_artist,
                )
                score, reasons, violations = score_candidate(
                    current, data, mode, controls, artist_run_length=run
                )
                if violations:
                    continue
                item = {**candidate, "score": score, "reasons": reasons}
                next_beams.append(
                    (total + score, [*path, item], [x for x in remaining if x is not candidate])
                )
        next_beams.sort(key=lambda x: x[0], reverse=True)
        beams = next_beams[:beam_width]
        if not beams:
            break
    return beams[0][1] if beams else []


def _optimize_segment(
    segment: list[dict[str, Any]],
    anchor: dict[str, Any] | None,
    mode: DJMode,
    controls: DJControls,
    beam_width: int,
) -> list[dict[str, Any]]:
    anchor_artist = anchor.get("artist") if anchor else None
    optimized = _beam_pass(segment, anchor, mode, controls, beam_width, anchor_artist)
    if len(optimized) != len(segment) and anchor_artist is not None:
        # The anchor boundary can make the artist-run constraint unsatisfiable
        # (e.g. the queue continues the now-playing artist): no ordering of the
        # segment can satisfy it, and bailing out fails the whole queue. The
        # queue is fixed reality; the optimizer only chooses the order. Retry
        # with the boundary constraint relaxed rather than brick the queue.
        optimized = _beam_pass(segment, anchor, mode, controls, beam_width, None)
    return optimized


def beam_optimize(  # noqa: PLR0915 - placement pipeline reads best as one pass
    tracks: list[dict[str, Any]],
    current: dict[str, Any] | None,
    mode: DJMode,
    *,
    controls: DJControls | None = None,
    beam_width: int = 12,
    fixed_ids: set[str] | None = None,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Optimize all movable tracks while preserving fixed positions and hard requirements."""
    controls = controls or DJControls()
    fixed_ids = set(fixed_ids or controls.fixed_ids)
    excluded_ids = set(excluded_ids or controls.excluded_ids)

    ids = {str(t.get("queue_item_id")) for t in tracks}
    missing_required = set(controls.required_ids) - ids
    if missing_required:
        raise RuntimeError(f"Required tracks are missing: {sorted(missing_required)}")
    conflict = controls.required_ids & controls.excluded_ids
    if conflict:
        raise RuntimeError(f"Tracks cannot be both required and excluded: {sorted(conflict)}")
    if controls.end_track_id and controls.end_track_id in controls.excluded_ids:
        raise RuntimeError("End track cannot also be excluded")
    if controls.end_track_id and controls.end_track_id not in ids:
        raise RuntimeError(f"End track is missing: {controls.end_track_id}")

    fixed_positions = {
        idx: t
        for idx, t in enumerate(tracks)
        if t.get("queue_item_id") in fixed_ids
        or (not isinstance(t.get("analysis"), dict) and t.get("queue_item_id") not in excluded_ids)
    }

    # Optimize each interval between fixed anchors. Excluded tracks do not occupy
    # an output slot, so they must not affect segment sizing.
    result: list[dict[str, Any]] = []
    fixed_indices = sorted(fixed_positions)
    interval_starts = [0, *[idx + 1 for idx in fixed_indices]]
    interval_ends = [*fixed_indices, len(tracks)]
    for start, end in zip(interval_starts, interval_ends, strict=True):
        segment = [
            track
            for idx, track in enumerate(tracks[start:end], start=start)
            if idx not in fixed_positions and track.get("queue_item_id") not in excluded_ids
        ]
        if segment:
            anchor = (
                None
                if result and not isinstance(result[-1].get("analysis"), dict)
                else (_merge_track(result[-1]) if result else current)
            )
            optimized = _optimize_segment(segment, anchor, mode, controls, beam_width)
            if len(optimized) != len(segment):
                raise RuntimeError("Hard requirements are impossible with the current queue")
            result.extend(optimized)
        if end < len(tracks):
            fixed = fixed_positions[end]
            if fixed.get("queue_item_id") in controls.excluded_ids:
                raise RuntimeError("A fixed track is excluded")
            fixed_data = _merge_track(fixed)
            reason = (
                "analysis unavailable; preserved"
                if not isinstance(fixed.get("analysis"), dict)
                else "fixed track"
            )
            result.append(
                {
                    **fixed,
                    "score": None if reason.startswith("analysis") else 1.0,
                    "reasons": [reason],
                }
            )
            current = None if reason.startswith("analysis") else fixed_data

    # Excluded tracks are intentionally absent; every other non-fixed track must remain exactly once.
    expected = {
        str(t.get("queue_item_id")) for t in tracks if t.get("queue_item_id") not in excluded_ids
    }
    actual = {str(t.get("queue_item_id")) for t in result}
    if expected != actual:
        raise RuntimeError("Optimizer would lose or duplicate queue tracks")

    required_present = controls.required_ids <= actual
    if not required_present:
        raise RuntimeError("Hard required tracks were not placed")

    if controls.end_track_id:
        end_index = next(
            i for i, item in enumerate(result) if item.get("queue_item_id") == controls.end_track_id
        )
        if end_index != len(result) - 1:
            if result[end_index].get("queue_item_id") in fixed_ids:
                raise RuntimeError("End track is fixed and cannot be moved to the end")
            end_item = result.pop(end_index)
            result.append({**end_item, "reasons": [*end_item.get("reasons", []), "end track"]})

    annotated: list[dict[str, Any]] = []
    previous = current
    for item in result:
        analysis = item.get("analysis") if isinstance(item.get("analysis"), dict) else None
        bpm_change = None
        energy_delta = None
        key_affinity = None
        if isinstance(previous, dict) and isinstance(analysis, dict):
            if isinstance(previous.get("bpm"), (int, float)) and isinstance(
                analysis.get("bpm"), (int, float)
            ):
                bpm_change = float(analysis["bpm"]) - float(previous["bpm"])
            if isinstance(previous.get("energy"), (int, float)) and isinstance(
                analysis.get("energy"), (int, float)
            ):
                energy_delta = float(analysis["energy"]) - float(previous["energy"])
            key_affinity = camelot_affinity(previous.get("camelot"), analysis.get("camelot"))
        annotated.append(
            {
                **item,
                "bpm_change": bpm_change,
                "energy_delta": energy_delta,
                "key_affinity": key_affinity,
                "clap_similarity": (
                    clap_similarity(previous.get("clap_embedding"), analysis.get("clap_embedding"))
                    if isinstance(previous, dict) and isinstance(analysis, dict)
                    else None
                ),
                "transition_bars": controls.transition_bars_overrides.get(
                    str(item.get("queue_item_id")), controls.transition_bars
                ),
            }
        )
        previous = analysis
    return annotated
