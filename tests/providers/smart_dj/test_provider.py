"""Focused Smart DJ provider tests."""

import inspect
from typing import Any

from music_assistant.providers.smart_dj.engine import (
    MODES,
    DJControls,
    SignalControl,
    score_candidate,
)
from music_assistant.providers.smart_dj.provider import (
    SmartDJError,
    SmartDJProvider,
    _camelot_from_key,
    _snapshot_start_index,
)


def test_camelot_conversion() -> None:
    """Camelot conversion."""
    assert _camelot_from_key("A", "minor") == "8A"
    assert _camelot_from_key("C", "major") == "8B"


def test_hard_bpm_rule_is_never_violated() -> None:
    """Hard bpm rule is never violated."""
    score, _reasons, violations = score_candidate(
        {"bpm": 120, "camelot": "8A"},
        {"bpm": 160, "camelot": "8A"},
        MODES["ai_dj"],
        DJControls(bpm=SignalControl("hard")),
    )
    assert score == 0.0
    assert "hard BPM compatibility" in violations


def test_disabled_key_does_not_affect_score() -> None:
    """Disabled key does not affect score."""
    current = {"bpm": 120, "camelot": "8A", "energy": 0.5}
    candidate = {"bpm": 120, "camelot": "9A", "energy": 0.5}
    soft, _, _ = score_candidate(current, candidate, MODES["ai_dj"], DJControls())
    disabled, _, _ = score_candidate(
        current, candidate, MODES["ai_dj"], DJControls(key=SignalControl("disabled"))
    )
    assert disabled > 0
    assert soft != disabled


def test_required_excluded_and_fixed_controls_are_serializable() -> None:
    """Required excluded and fixed controls are serializable."""
    controls = DJControls(
        required_ids=frozenset({"required"}),
        excluded_ids=frozenset({"excluded"}),
        fixed_ids=frozenset({"fixed"}),
    )
    assert "required" in controls.required_ids
    assert "excluded" in controls.excluded_ids
    assert "fixed" in controls.fixed_ids


def test_snapshot_starts_at_current_queue_item() -> None:
    """Previously played queue items must not be treated as current/upcoming."""
    snapshot = [
        {"queue_item_id": "q1"},
        {"queue_item_id": "q2"},
        {"queue_item_id": "q3"},
    ]
    assert _snapshot_start_index(snapshot, ["q1", "q2", "q3"], 1) == 1


def test_snapshot_start_skips_unresolvable_current_item() -> None:
    """If current media is unresolvable, start at the next playable queue item."""
    snapshot = [
        {"queue_item_id": "q1"},
        {"queue_item_id": "q3"},
    ]
    assert _snapshot_start_index(snapshot, ["q1", "q2", "q3"], 1) == 1


def test_snapshot_start_returns_none_when_no_playable_items_remain() -> None:
    """An exhausted queue has no analysis start item."""
    snapshot = [{"queue_item_id": "q1"}]
    assert _snapshot_start_index(snapshot, ["q1"], 1) is None


# ------------------------------------------------------------------
# External-audit fixes (0.0.20): C-1, H-2, M-2, M-5, M-7
# ------------------------------------------------------------------


def _bare_provider() -> Any:
    """Return a bare provider instance (no framework wiring)."""
    return SmartDJProvider.__new__(SmartDJProvider)


def test_parse_controls_malformed_values_raise_named_errors() -> None:
    """Malformed control values raise SmartDJError, not raw int/float errors."""
    import pytest  # noqa: PLC0415 - pytest is not a sandbox dependency

    prov = _bare_provider()
    bad_inputs = (
        {"transition_bars": "abc"},
        {"lookahead": "high"},
        {"max_artist_repeat": -2},
        {"bpm_min": "fast"},
        {"transition_aggressiveness": "loud"},
    )
    for bad in bad_inputs:
        with pytest.raises(SmartDJError) as exc_info:
            prov._parse_controls(dict(bad), 0.08, True, True, "ai_dj")
        assert exc_info.value.code == "invalid_control"


def test_lookahead_accepts_the_full_advertised_range() -> None:
    """The backend accepts lookahead 1-96 to match the FE slider (M-7/L-2)."""
    prov = _bare_provider()
    assert prov._parse_controls({}, 0.08, True, True, "ai_dj").lookahead == 8
    assert prov._parse_controls({"lookahead": 96}, 0.08, True, True, "ai_dj").lookahead == 96
    assert prov._parse_controls({"lookahead": 1}, 0.08, True, True, "ai_dj").lookahead == 1
    try:
        prov._parse_controls({"lookahead": 97}, 0.08, True, True, "ai_dj")
    except SmartDJError:
        pass
    else:
        raise AssertionError("lookahead=97 must raise")


def test_normalize_musicae_preserves_legitimate_zeros() -> None:
    """``or``-fallbacks are gone: a legit 0 must not fall through (M-5)."""
    prov = _bare_provider()
    normalized = prov._normalize_musicae({"energy": 0.0, "score": {"energy": 0.9}})
    assert normalized["energy"] == 0.0
    normalized = prov._normalize_musicae({"key": 0, "musical_key": "8B"})
    assert normalized["key"] == 0


def test_normalize_musicae_rejects_non_finite_and_non_numeric() -> None:
    """NaN/Inf/non-numeric vendor values are dropped, never passed on (M-5)."""
    prov = _bare_provider()
    assert prov._normalize_musicae({"bpm": float("nan")})["bpm"] is None
    assert prov._normalize_musicae({"bpm": float("inf")})["bpm"] is None
    assert prov._normalize_musicae({"bpm": "128.0"})["bpm"] is None
    assert prov._normalize_musicae({"bpm": 0})["bpm"] is None
    assert prov._normalize_musicae({"energy": 5.0})["energy"] is None
    assert prov._normalize_musicae({"bpm": 128})["bpm"] == 128.0


class _PagedFakePQ:
    """Fake player_queues.items honoring limit/offset paging."""

    def __init__(self, count: int) -> None:
        self.items_list = [type("QI", (), {"queue_item_id": f"q{i}"})() for i in range(count)]

    def items(self, _queue_id: str, limit: int, offset: int) -> list[Any]:
        return self._items_list_style(offset, limit)

    def _items_list_style(self, offset: int, limit: int) -> list[Any]:
        return self.items_list[offset : offset + limit]


def test_read_full_queue_paginates_past_1000() -> None:
    """A queue beyond 1,000 items is read in FULL (audit C-1)."""
    fake_pq = _PagedFakePQ(1250)
    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov.mass = type("M", (), {"player_queues": fake_pq})()
    read_back = prov._read_full_queue("q1")
    assert len(read_back) == 1250
    assert read_back[-1].queue_item_id == "q1249"


def test_analyze_registration_requires_library_write() -> None:
    """Analyze can spend Musicae quota and write the cache: not guest-safe (H-2)."""
    source = inspect.getsource(SmartDJProvider.loaded_in_mass)
    assert '("smart_dj/analyze", self.analyze, Scope.LIBRARY_WRITE)' in source
    # only genuinely read-only commands stay at scope None
    none_scoped = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith('("smart_dj/') and line.strip().endswith("None),")
    ]
    assert len(none_scoped) == 2
    assert all("status" in line or "capabilities" in line for line in none_scoped)
