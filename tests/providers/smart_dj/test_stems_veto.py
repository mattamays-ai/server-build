"""
Tests for the 0.0.28 lane D per-transition stems veto (server half).

Covers the plan-time veto derivation (_stems_veto_reason), its metadata
exposure through _transition_report (stems_used / stems_veto_reason), and
the _automix_stems_vetoes pin walk. The playback-side mask floor lives in
tests/controllers/streams/smart_fades/test_stems.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant.controllers.streams.smart_fades import stems as stems_ducking
from music_assistant.providers.smart_dj.engine import DJControls
from music_assistant.providers.smart_dj.provider import (
    STEMS_VETO_DISABLED,
    STEMS_VETO_OUTGOING_INSTRUMENTAL,
    STEMS_VETO_OUTGOING_VOCAL_DOMINANCE,
    STEMS_VETO_UNAVAILABLE,
    _automix_stems_vetoes,
    _stems_veto_reason,
    _transition_report,
)

if TYPE_CHECKING:
    import pytest


def _enable_stems(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the separation stack to look present (sandbox has no model)."""
    monkeypatch.setattr(stems_ducking, "available", lambda: True)


def test_veto_when_toggle_off() -> None:
    """Default behavior: with the user's stems toggle off the veto says so."""
    assert _stems_veto_reason({"name": "A"}, False) == STEMS_VETO_DISABLED


def test_veto_when_stack_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """No separation runtime/model: the duck could never apply."""
    monkeypatch.setattr(stems_ducking, "available", lambda: False)
    assert _stems_veto_reason({"name": "A"}, True) == STEMS_VETO_UNAVAILABLE


def test_veto_outgoing_instrumental_by_density(monkeypatch: pytest.MonkeyPatch) -> None:
    """An outgoing deck under the near-instrumental duty line has nothing to duck."""
    _enable_stems(monkeypatch)
    assert _stems_veto_reason({"vocal_density": 0.01}, True) == STEMS_VETO_OUTGOING_INSTRUMENTAL


def test_veto_outgoing_instrumental_by_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a density the analysis instrumental claim is honored."""
    _enable_stems(monkeypatch)
    assert _stems_veto_reason({"instrumental": True}, True) == STEMS_VETO_OUTGOING_INSTRUMENTAL


def test_veto_outgoing_vocal_dominance(monkeypatch: pytest.MonkeyPatch) -> None:
    """At or past the index's vocal tier-2 line the duck is vetoed."""
    _enable_stems(monkeypatch)
    assert _stems_veto_reason({"vocal_density": 0.30}, True) == STEMS_VETO_OUTGOING_VOCAL_DOMINANCE


def test_no_veto_on_a_balanced_outgoing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Between the two lines the duck runs (reason None)."""
    _enable_stems(monkeypatch)
    assert _stems_veto_reason({"vocal_density": 0.10}, True) is None


def test_veto_thresholds_are_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary densities: 0.05 is NOT instrumental, 0.25 IS dominance."""
    _enable_stems(monkeypatch)
    assert _stems_veto_reason({"vocal_density": 0.05}, True) is None
    assert _stems_veto_reason({"vocal_density": 0.25}, True) == STEMS_VETO_OUTGOING_VOCAL_DOMINANCE


def test_veto_reads_nested_analysis_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queue/playlist snapshot rows carry their analysis nested."""
    _enable_stems(monkeypatch)
    assert (
        _stems_veto_reason({"analysis": {"vocal_density": 0.30}}, True)
        == STEMS_VETO_OUTGOING_VOCAL_DOMINANCE
    )


def test_veto_unknown_density_without_instrumental_claim_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No measurement is not evidence of low confidence: the duck may run."""
    _enable_stems(monkeypatch)
    assert _stems_veto_reason({"name": "A"}, True) is None


def test_transition_report_carries_stems_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each note exposes stems_used + stems_veto_reason for the FE."""
    _enable_stems(monkeypatch)
    control = DJControls(stems_enabled=True)
    ordered = [
        {"name": "A", "vocal_density": 0.30},
        {"name": "B", "vocal_density": 0.10},
        {"name": "C", "vocal_density": 0.01},
        {"name": "D", "vocal_density": 0.10},
    ]
    notes = _transition_report(ordered, control)
    assert notes[0]["stems_used"] is False
    assert notes[0]["stems_veto_reason"] == STEMS_VETO_OUTGOING_VOCAL_DOMINANCE
    assert notes[1]["stems_used"] is True
    assert notes[1]["stems_veto_reason"] is None
    # the blend INTO D reads D's OUTGOING predecessor (C, instrumental): veto
    assert notes[2]["stems_used"] is False
    assert notes[2]["stems_veto_reason"] == STEMS_VETO_OUTGOING_INSTRUMENTAL


def test_transition_report_toggle_off_reports_disabled() -> None:
    """With the toggle off every note reports stems_disabled (the default)."""
    control = DJControls()
    notes = _transition_report([{"name": "A"}, {"name": "B"}], control)
    assert notes[0]["stems_used"] is False
    assert notes[0]["stems_veto_reason"] == STEMS_VETO_DISABLED


def test_automix_stems_vetoes_mirror_blend_pref_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same walk/keys as _automix_blend_prefs: the blend INTO a track reads it."""
    _enable_stems(monkeypatch)
    control = DJControls(stems_enabled=True)
    tracks: list[dict[str, Any]] = [
        {"queue_item_id": "q0", "vocal_density": 0.10},
        {"queue_item_id": "q1", "vocal_density": 0.01},
        {"queue_item_id": "q2", "vocal_density": 0.30},
        {"name": "no-id row is skipped like the blend prefs"},
    ]
    vetoes = _automix_stems_vetoes(tracks, control)
    # q0: no known outgoing -> no veto; q1: outgoing q0 balanced -> no veto;
    # q2: outgoing q1 is instrumental -> veto pinned to q2 (the blend INTO q2
    # reads q1's outgoing side). The no-id row is skipped without disturbing
    # the walk.
    assert vetoes == {"q2": STEMS_VETO_OUTGOING_INSTRUMENTAL}


def test_automix_stems_vetoes_read_outgoing_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dominance veto pins to the blend INTO the FOLLOWING track."""
    _enable_stems(monkeypatch)
    control = DJControls(stems_enabled=True)
    tracks = [
        {"queue_item_id": "q0", "vocal_density": 0.30},
        {"queue_item_id": "q1", "vocal_density": 0.30},
    ]
    vetoes = _automix_stems_vetoes(tracks, control)
    assert vetoes == {"q1": STEMS_VETO_OUTGOING_VOCAL_DOMINANCE}


def test_automix_stems_vetoes_omit_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """stems_disabled never pins a per-track veto: the global toggle covers it."""
    monkeypatch.setattr(stems_ducking, "available", lambda: True)
    vetoes = _automix_stems_vetoes([{"queue_item_id": "q1"}], DJControls(stems_enabled=False))
    assert vetoes == {}


def test_automix_stems_vetoes_previous_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first entry's outgoing can be handed over explicitly."""
    _enable_stems(monkeypatch)
    control = DJControls(stems_enabled=True)
    vetoes = _automix_stems_vetoes(
        [{"queue_item_id": "q1", "vocal_density": 0.10}],
        control,
        previous={"vocal_density": 0.30},
    )
    assert vetoes == {"q1": STEMS_VETO_OUTGOING_VOCAL_DOMINANCE}
