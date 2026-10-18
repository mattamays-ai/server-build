"""Tests for the 0.0.27 FLOATING energy_arc control (scoring-only asymmetry)."""

from __future__ import annotations

from music_assistant.providers.smart_dj.engine import (
    ARC_CLIMB_MAX,
    ARC_RESET_DROP,
    MODES,
    DJControls,
    SignalControl,
    _arc_reset_count,
    _arc_step,
    _signal_values,
    beam_optimize,
    score_candidate,
)

AI = MODES["ai_dj"]


def test_arc_step_climbs_are_capped() -> None:
    """A climb past ~1 tier holds the target at the cap."""
    step, reason = _arc_step(0.5, 0.9, AI.energy_direction, position=3, segment_length=6, resets=0)
    assert step == ARC_CLIMB_MAX
    assert reason == "arc: hold"


def test_arc_step_reset_when_available() -> None:
    """A 2-3 tier drop is accepted as a reset: the target follows the drop."""
    step, reason = _arc_step(0.8, 0.3, AI.energy_direction, position=3, segment_length=6, resets=0)
    assert reason == "arc: reset"
    assert step == 0.3 - 0.8  # the target rides all the way down


def test_arc_step_reset_restrained_once_spent() -> None:
    """Max one reset per segment: afterwards big drops are held, not rewarded."""
    step, reason = _arc_step(0.8, 0.3, AI.energy_direction, position=3, segment_length=6, resets=1)
    assert reason == "arc: hold"
    assert step == -ARC_CLIMB_MAX


def test_arc_step_warmup_restraint() -> None:
    """Early in the segment the climb cap is halved."""
    step, reason = _arc_step(0.5, 0.9, AI.energy_direction, position=0, segment_length=9, resets=0)
    assert step == ARC_CLIMB_MAX / 2
    assert reason == "arc: hold"


def test_arc_step_lift() -> None:
    """A climb within the cap along the mode's direction lifts."""
    step, reason = _arc_step(0.5, 0.6, AI.energy_direction, position=3, segment_length=6, resets=0)
    assert reason == "arc: lift"
    assert step == AI.energy_direction


def test_arc_off_by_default() -> None:
    """Without the control there are NO arc reasons and identical scoring."""
    current = {"bpm": 120.0, "energy": 0.5, "camelot": "8A"}
    candidate = {"bpm": 121.0, "energy": 0.9, "camelot": "8A"}
    off_score, off_reasons = _signal_values(
        current, candidate, AI, DJControls(), artist_run_length=0
    )
    arc_off_score, arc_off_reasons = _signal_values(
        current, candidate, AI, DJControls(energy_arc=False), artist_run_length=0
    )
    assert off_score == arc_off_score
    assert off_reasons == arc_off_reasons
    assert not any(str(r).startswith("arc:") for r in off_reasons)


def test_arc_reasons_surface_in_signal_values() -> None:
    """'arc: lift/hold/reset' rides in reasons so the FE can show why."""
    current = {"bpm": 120.0, "energy": 0.5, "camelot": "8A"}
    climb = {"bpm": 121.0, "energy": 0.6, "camelot": "8A"}
    overshoot = {"bpm": 121.0, "energy": 0.9, "camelot": "8A"}
    reset = {"bpm": 121.0, "energy": 0.1, "camelot": "8A"}
    control = DJControls(energy_arc=True)
    _, lift = _signal_values(
        current,
        climb,
        AI,
        control,
        artist_run_length=0,
        position=3,
        segment_length=6,
        arc_resets=0,
    )
    _, hold = _signal_values(
        current,
        overshoot,
        AI,
        control,
        artist_run_length=0,
        position=3,
        segment_length=6,
        arc_resets=0,
    )
    _, reset_r = _signal_values(
        current,
        reset,
        AI,
        control,
        artist_run_length=0,
        position=3,
        segment_length=6,
        arc_resets=0,
    )
    assert "arc: lift" in lift
    assert "arc: hold" in hold
    assert "arc: reset" in reset_r


def test_arc_prefers_the_cap_respecting_climb() -> None:
    """A climb within the cap must outscore an overshooting climb."""
    current = {"bpm": 120.0, "energy": 0.5, "camelot": "8A"}
    control = DJControls(energy_arc=True)
    within, _, _ = score_candidate(
        current,
        {"bpm": 120.0, "energy": 0.5 + ARC_CLIMB_MAX, "camelot": "8A"},
        AI,
        control,
        position=3,
        segment_length=6,
        arc_resets=0,
    )
    over, _, _ = score_candidate(
        current,
        {"bpm": 120.0, "energy": 0.5 + 2 * ARC_CLIMB_MAX, "camelot": "8A"},
        AI,
        control,
        position=3,
        segment_length=6,
        arc_resets=0,
    )
    assert within > over


def test_hard_gate_stays_symmetric_under_arc() -> None:
    """The arc shapes SCORING only: a hard energy delta still hard-fails."""
    current = {"bpm": 120.0, "energy": 0.5, "camelot": "8A"}
    control = DJControls(energy_arc=True, energy=SignalControl("hard"))
    score, _, violations = score_candidate(
        current,
        {"bpm": 120.0, "energy": 0.5 - ARC_RESET_DROP, "camelot": "8A"},
        AI,
        control,
        position=3,
        segment_length=6,
        arc_resets=0,
    )
    assert score == 0.0
    assert violations  # the reset is NOT waved through the hard gate


def test_arc_reset_count_walks_the_path() -> None:
    """Resets already on a path (anchor included) are counted."""
    anchor = {"energy": 0.9}  # anchors arrive already merged
    path = [
        {"analysis": {"energy": 0.4}},  # drop 0.5 from the anchor: reset 1
        {"analysis": {"energy": 0.45}},
        {"analysis": {"energy": 0.0}},  # drop 0.45: reset 2
    ]
    assert _arc_reset_count(anchor, path) == 2
    assert _arc_reset_count(anchor, path[:2]) == 1
    assert _arc_reset_count(None, []) == 0


def test_beam_optimize_runs_with_arc_enabled() -> None:
    """beam_optimize with the control on: full order, no crash, no veto change."""
    tracks = [
        {
            "queue_item_id": qid,
            "artist": f"a-{qid}",
            "analysis": {"bpm": bpm, "energy": energy, "camelot": "8A"},
        }
        for qid, bpm, energy in (("t1", 120.0, 0.2), ("t2", 121.0, 0.8), ("t3", 122.0, 0.5))
    ]
    control = DJControls(energy_arc=True)
    optimized = beam_optimize(tracks, None, AI, controls=control)
    assert len(optimized) == 3
    assert {t["queue_item_id"] for t in optimized} == {"t1", "t2", "t3"}
