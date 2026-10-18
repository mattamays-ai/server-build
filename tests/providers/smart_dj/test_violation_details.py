"""Tests for the hard-requirements insight details (0.0.27 spec Item 2)."""

from __future__ import annotations

from typing import Any

import pytest

from music_assistant.providers.smart_dj.engine import (
    MODES,
    DJControls,
    HardRequirementsImpossible,
    SignalControl,
    _merge_track,
    _violation_details,
    beam_optimize,
    score_candidate,
)


def _track(**overrides: Any) -> dict[str, Any]:
    """Build a complete analysis dict with sensible defaults."""
    track: dict[str, Any] = {
        "bpm": 120.0,
        "camelot": "8A",
        "energy": 0.5,
        "danceability": 0.5,
        "loudness": -10.0,
        "genre": "house",
        "artist": "A",
    }
    track.update(overrides)
    return track


def _capture(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    controls: DJControls,
    violations: list[str],
    artist_run_length: int = 0,
) -> dict[str, Any]:
    """Build the best-rejected capture record exactly as _beam_pass threads it."""
    return {
        "rank": 0.0,
        "current": current,
        "candidate": candidate,
        "controls": controls,
        "violations": violations,
        "artist_run_length": artist_run_length,
    }


def _violations(current: dict[str, Any], candidate: dict[str, Any], controls: DJControls) -> list[str]:
    """Run the REAL gate to get the violation strings for a pair."""
    _, _, violations = score_candidate(current, candidate, MODES["ai_dj"], controls)
    return violations


# ---------------------------------------------------------------------------
# Blocked case: rules with real numbers
# ---------------------------------------------------------------------------


def test_blocked_energy_pair_reports_real_numbers() -> None:
    """A hard energy violation re-derives the 0-1 delta against the 0.35 gate."""
    current, candidate = _track(), _track(energy=0.9)
    controls = DJControls(energy=SignalControl("hard", 1.0))
    violations = _violations(current, candidate, controls)
    assert violations == ["hard energy compatibility"]

    details = _violation_details(_capture(current, candidate, controls, violations))
    assert len(details) == 1
    row = details[0]
    assert row["rule"] == "energy_step"
    assert row["actual"] == pytest.approx(0.4)
    assert row["limit"] == pytest.approx(0.35)
    assert row["unit"] == "energy (0-1)"
    # the full panel contract: every row carries the suggestion slots
    for key in ("rule", "actual", "limit", "unit", "suggested_setting", "suggested_value"):
        assert key in row


def test_blocked_bpm_stretch_pair_reports_percent_of_anchor() -> None:
    """The stretch gate reports actual/limit as percent of the anchor BPM."""
    current = _track(bpm=120.0, instrumental=True)
    candidate = _track(bpm=130.0, instrumental=True)
    controls = DJControls(bpm=SignalControl("hard", 1.0), beat_strictness="strict")
    violations = _violations(current, candidate, controls)
    assert violations == ["hard BPM compatibility"]

    details = _violation_details(_capture(current, candidate, controls, violations))
    assert len(details) == 1
    row = details[0]
    assert row["rule"] == "bpm_delta"
    assert row["actual"] == pytest.approx(8.33, abs=0.01)  # 10/120 as %
    assert row["limit"] == pytest.approx(6.0)  # strict = 6% of anchor, unhalved
    assert row["unit"] == "% of anchor bpm"
    assert "vocal_halved" not in row


def test_bpm_suggestion_targets_beat_strictness_with_next_looser_value() -> None:
    """The fix suggestion writes beatStrictness, the knob the gate actually reads."""
    current = _track(bpm=120.0, instrumental=True)
    candidate = _track(bpm=128.0, instrumental=True)
    controls = DJControls(bpm=SignalControl("hard", 1.0), beat_strictness="strict")
    violations = _violations(current, candidate, controls)
    assert violations == ["hard BPM compatibility"]

    details = _violation_details(_capture(current, candidate, controls, violations))
    row = details[0]
    assert row["suggested_setting"] == "beatStrictness"
    # 6.67% actual: strict (6%) cannot cover it, normal (8%) can
    assert row["suggested_value"] == "normal"


def test_bpm_suggestion_is_none_when_even_aggressive_cannot_cover() -> None:
    """Past every strictness ceiling the honest suggestion is swap tracks (None)."""
    current, candidate = _track(bpm=120.0), _track(bpm=150.0)
    controls = DJControls(bpm=SignalControl("hard", 1.0), beat_strictness="strict")
    violations = _violations(current, candidate, controls)
    assert violations == ["hard BPM compatibility"]

    details = _violation_details(_capture(current, candidate, controls, violations))
    row = details[0]
    assert row["suggested_setting"] == "beatStrictness"
    assert row["suggested_value"] is None


def test_max_bpm_jump_reports_absolute_bpm() -> None:
    """The max-jump rule reports BPM units, not percent."""
    current, candidate = _track(bpm=120.0), _track(bpm=140.0)
    controls = DJControls(max_bpm_jump=10.0)
    violations = _violations(current, candidate, controls)
    assert violations == ["maximum BPM jump exceeded"]

    details = _violation_details(_capture(current, candidate, controls, violations))
    row = details[0]
    assert row["rule"] == "bpm_delta"
    assert row["actual"] == pytest.approx(20.0)
    assert row["limit"] == pytest.approx(10.0)
    assert row["unit"] == "bpm"


def test_key_row_reports_camelot_affinity_not_steps() -> None:
    """The key gate reports the affinity value plus both Camelot codes."""
    current, candidate = _track(camelot="8A"), _track(camelot="3A")
    controls = DJControls(key=SignalControl("hard", 1.0))
    violations = _violations(current, candidate, controls)
    # the compatible key_relation veto AND the hard gate both fire for this pair
    assert violations == ["incompatible key", "hard key compatibility unavailable or failed"]

    details = _violation_details(_capture(current, candidate, controls, violations))
    assert len(details) == 2
    assert {row["rule"] for row in details} == {"key_incompatible"}
    row = details[0]
    assert row["actual"] == pytest.approx(0.0)  # 8A vs 3A affinity tier
    assert row["unit"] == "camelot affinity"
    assert row["detail"] == "8A -> 3A"


def test_artist_spacing_row_reports_run_vs_cap() -> None:
    """The spacing rule reports the consecutive-run length against the cap."""
    current, candidate = _track(artist="A"), _track(artist="A")
    controls = DJControls(artist_spacing=SignalControl("hard", 1.0))
    violations = ["hard artist spacing"]

    details = _violation_details(
        _capture(current, candidate, controls, violations, artist_run_length=1)
    )
    row = details[0]
    assert row["rule"] == "artist_spacing"
    assert row["actual"] == 1
    assert row["limit"] == 1  # max(1, max_artist_repeat) with the default cap
    assert row["unit"] == "consecutive tracks"


def test_informational_row_keeps_the_raw_violation_string() -> None:
    """Violations with no derivable numbers render as informational rows."""
    details = _violation_details(_capture(_track(), _track(), DJControls(), ["excluded track"]))
    assert len(details) == 1
    assert details[0]["rule"] == "excluded track"
    assert details[0]["actual"] is None
    assert details[0]["limit"] is None


def test_rows_sort_by_overshoot_times_weight() -> None:
    """The most-overshooting rule family leads (bpm weight 1.0 > energy 0.6)."""
    current = _track(bpm=120.0, energy=0.5)
    candidate = _track(bpm=150.0, energy=0.9)
    controls = DJControls(
        bpm=SignalControl("hard", 1.0),
        energy=SignalControl("hard", 1.0),
        beat_strictness="strict",
    )
    violations = _violations(current, candidate, controls)
    assert set(violations) == {"hard BPM compatibility", "hard energy compatibility"}

    details = _violation_details(_capture(current, candidate, controls, violations))
    assert [row["rule"] for row in details] == ["bpm_delta", "energy_step"]


# ---------------------------------------------------------------------------
# Clean case: empty list
# ---------------------------------------------------------------------------


def test_clean_plan_returns_empty_list() -> None:
    """No capture (a clean plan) yields no rows - never a fake rule."""
    assert _violation_details(None) == []
    assert _violation_details(_capture(_track(), _track(), DJControls(), [])) == []


def test_clean_pair_produces_no_rows_through_the_real_gate() -> None:
    """A compatible pair passes the real gate and the details stay empty."""
    current, candidate = _track(), _track()
    controls = DJControls(bpm=SignalControl("hard", 1.0), energy=SignalControl("hard", 1.0))
    assert _violations(current, candidate, controls) == []
    assert _violation_details(_capture(current, candidate, controls, [])) == []


# ---------------------------------------------------------------------------
# Vocal halving
# ---------------------------------------------------------------------------


def test_vocal_halving_is_reflected_in_the_reported_limit() -> None:
    """The stretch window halves when either track is vocal, and the row shows it."""
    current, candidate = _track(bpm=120.0, instrumental=False), _track(bpm=126.0, instrumental=False)
    controls = DJControls(bpm=SignalControl("hard", 1.0), beat_strictness="normal")
    violations = _violations(current, candidate, controls)
    # 5% actual: passes the plain 8% window, fails the vocal-halved 4% window
    assert violations == ["hard BPM compatibility"]

    details = _violation_details(_capture(current, candidate, controls, violations))
    row = details[0]
    assert row["limit"] == pytest.approx(4.0)  # normal 8% halved
    assert row["vocal_halved"] is True
    # 5% fits the vocal-halved aggressive window (6%)
    assert row["suggested_value"] == "aggressive"


def test_instrumental_pair_is_not_halved() -> None:
    """Two instrumental tracks keep the full window - the halving must not leak."""
    current, candidate = _track(bpm=120.0, instrumental=True), _track(bpm=126.0, instrumental=True)
    controls = DJControls(bpm=SignalControl("hard", 1.0), beat_strictness="strict")
    # 5% actual vs the plain strict 6% window: passes, so nothing to report
    assert _violations(current, candidate, controls) == []


# ---------------------------------------------------------------------------
# End-to-end: beam_optimize threads the details out on the failure path
# ---------------------------------------------------------------------------


def test_beam_optimize_raise_carries_violation_details() -> None:
    """An unsatisfiable hard constraint raises with re-derived panel rows."""
    tracks = [
        {"queue_item_id": "a", "artist": "a", "analysis": _track(energy=0.0)},
        {"queue_item_id": "b", "artist": "b", "analysis": _track(energy=0.9)},
    ]
    controls = DJControls(energy=SignalControl("hard", 1.0))
    with pytest.raises(HardRequirementsImpossible) as exc_info:
        beam_optimize(tracks, None, MODES["ai_dj"], controls=controls)
    # still a RuntimeError: every existing except RuntimeError keeps working
    assert isinstance(exc_info.value, RuntimeError)
    details = exc_info.value.details
    assert len(details) == 1
    assert details[0]["rule"] == "energy_step"
    assert details[0]["actual"] == pytest.approx(0.9)
    assert details[0]["limit"] == pytest.approx(0.35)


def test_beam_optimize_clean_queue_raises_nothing() -> None:
    """A satisfiable queue still plans without the new exception."""
    tracks = [
        {"queue_item_id": "a", "artist": "a", "analysis": _track()},
        {"queue_item_id": "b", "artist": "b", "analysis": _track(camelot="8B")},
    ]
    result = beam_optimize(tracks, _track(), MODES["ai_dj"], beam_width=4)
    assert {t["queue_item_id"] for t in result} == {"a", "b"}


def test_capture_uses_merged_track_data_like_the_beam_does() -> None:
    """The capture carries merged analysis data, so numbers survive the merge."""
    item = {"queue_item_id": "b", "artist": "b", "analysis": _track(energy=0.9)}
    merged = _merge_track(item)
    assert merged["energy"] == pytest.approx(0.9)
    controls = DJControls(energy=SignalControl("hard", 1.0))
    violations = _violations(_track(), merged, controls)
    details = _violation_details(_capture(_track(), merged, controls, violations))
    assert details[0]["actual"] == pytest.approx(0.4)
