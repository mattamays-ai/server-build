"""Tests for the 0.0.27 pair-aware _pair_style resolver + blend prefs."""

from __future__ import annotations

import re
from typing import Any

from music_assistant.providers.smart_dj.engine import DJControls
from music_assistant.providers.smart_dj.provider import (
    RULE_BACKSPIN_NEVER_AUTO,
    RULE_DOUBLE_DROP_NON_DNB,
    RULE_QUICK_CUT_BLOCKED,
    RULE_TRANSITION_BARS_CAPPED,
    _automix_blend_prefs,
    _pair_bar_cap,
    _pair_style,
    _style_guard,
    _transition_report,
)


def test_incoming_genre_wins_the_pin() -> None:
    """With style=auto the incoming track's genre fills the pin (today's path)."""
    control = DJControls()
    resolved = _pair_style({"genre": "Hip-Hop"}, {"genre": "House"}, control)
    assert resolved["style"] == "full_blend"
    assert resolved["strictness"] == "normal"
    assert resolved["source"] == "genre"
    assert resolved["rules"] == []


def test_outgoing_genre_is_consulted_when_incoming_has_none() -> None:
    """The gap this fixes: a house -> klezmer boundary keeps the house blend."""
    control = DJControls()
    resolved = _pair_style({"genre": "House"}, {"genre": "Klezmer"}, control)
    assert resolved["style"] == "full_blend"
    assert resolved["strictness"] == "normal"
    assert resolved["source"] == "genre_outgoing"


def test_engine_default_when_neither_side_matches() -> None:
    """No profile on either side: the planner decides per pair at blend time."""
    control = DJControls()
    resolved = _pair_style({"genre": "Klezmer"}, {"genre": "Polka"}, control)
    assert resolved["style"] == "auto"
    assert resolved["source"] == "engine"


def test_user_pick_wins_and_is_never_vetoed() -> None:
    """An explicit user quick_cut over house material is never blocked."""
    control = DJControls(transition_style="quick_cut")
    resolved = _pair_style({"genre": "House"}, {"genre": "Techno"}, control)
    assert resolved["style"] == "quick_cut"
    assert resolved["source"] == "user"
    assert resolved["rules"] == []


def test_style_guard_quick_cut_blocked_over_four_on_floor() -> None:
    """quick_cut must never be an AUTO choice in house/techno/trance."""
    assert _style_guard("quick_cut", {"genre": "House"}, {"genre": "Deep House"}) == [
        RULE_QUICK_CUT_BLOCKED
    ]
    assert _style_guard("quick_cut", None, {"genre": "Techno"}) == [RULE_QUICK_CUT_BLOCKED]
    # pop/hip-hop/rock/funk: quick_cut is the natural default there
    assert _style_guard("quick_cut", {"genre": "Hip-Hop"}, {"genre": "Funk"}) == []


def test_style_guard_double_drop_is_dnb_only() -> None:
    """double-drop outside drum & bass is blocked with a stable rule id."""
    assert _style_guard("double_drop", {"genre": "House"}, {"genre": "Pop"}) == [
        RULE_DOUBLE_DROP_NON_DNB
    ]
    assert _style_guard("double_drop", {"genre": "Drum & Bass"}, {"genre": "Jungle"}) == []


def test_style_guard_backspin_never_auto() -> None:
    """Backspin is never an auto choice, for any pair."""
    assert _style_guard("backspin", {"genre": "Pop"}, {"genre": "Rock"}) == [
        RULE_BACKSPIN_NEVER_AUTO
    ]


def test_rule_ids_are_stable_snake_case() -> None:
    """Every emitted rule id is snake_case (the FE keys definitions to it)."""
    for rule in (
        RULE_QUICK_CUT_BLOCKED,
        RULE_DOUBLE_DROP_NON_DNB,
        RULE_BACKSPIN_NEVER_AUTO,
        RULE_TRANSITION_BARS_CAPPED,
    ):
        assert re.fullmatch(r"[a-z][a-z0-9_]*", rule), rule


def test_bars_clamped_for_pop_pairs() -> None:
    """Global 32 bars over a pop pair derives to the 8-bar living range."""
    control = DJControls(transition_bars=32)
    resolved = _pair_style({"genre": "Pop"}, {"genre": "Hip-Hop"}, control)
    assert resolved["bars"] == 8
    assert RULE_TRANSITION_BARS_CAPPED in resolved["rules"]


def test_bars_clamped_for_dnb_pairs() -> None:
    """DnB tops out at 8 bars per pair."""
    control = DJControls(transition_bars=16)
    resolved = _pair_style({"genre": "Drum and Bass"}, {"genre": "Drum and Bass"}, control)
    assert resolved["bars"] == 8
    assert RULE_TRANSITION_BARS_CAPPED in resolved["rules"]


def test_bars_untouched_for_house_pairs() -> None:
    """house/techno/trance run 32-bar blends: no clamp, no rule id."""
    control = DJControls(transition_bars=32)
    resolved = _pair_style({"genre": "House"}, {"genre": "Trance"}, control)
    assert resolved["bars"] == 32
    assert RULE_TRANSITION_BARS_CAPPED not in resolved["rules"]


def test_pair_bar_cap_takes_the_stricter_side() -> None:
    """hip-hop outgoing x house incoming clamps to the hip-hop ceiling."""
    assert _pair_bar_cap({"genre": "Hip-Hop"}, {"genre": "House"}) == 16
    assert _pair_bar_cap({"genre": "House"}, {"genre": "House"}) == 32
    assert _pair_bar_cap(None, {"genre": "Polka"}) is None


def test_transition_report_carries_rules_and_bars() -> None:
    """The report's notes expose rules + the genre-clamped bar count."""
    control = DJControls(transition_bars=32)
    ordered = [
        {"name": "A", "bpm": 124.0, "genre": "House"},
        {"name": "B", "bpm": 126.0, "genre": "Pop"},
    ]
    notes = _transition_report(ordered, control)
    assert len(notes) == 1
    note = notes[0]
    assert note["style"] == "radio"  # incoming pop profile fills the pin
    assert note["source"] == "genre"
    assert note["transition_bars"] == 8  # the pop-family phrase math derives 8
    assert isinstance(note["rules"], list)


def test_blend_prefs_are_pair_aware() -> None:
    """_automix_blend_prefs consults the outgoing predecessor."""
    control = DJControls()
    tracks = [
        {"queue_item_id": "q0", "genre": "House"},
        {"queue_item_id": "q1", "genre": "Klezmer"},
    ]
    prefs: dict[str, tuple[str, str]] = _automix_blend_prefs(tracks, control)
    # q0 has no known outgoing: single-sided read (house -> full_blend)
    assert prefs["q0"] == ("full_blend", "normal")
    # q1's own genre has no profile, so the OUTGOING house side fills the pin
    assert prefs["q1"] == ("full_blend", "normal")


def test_blend_prefs_previous_kwarg() -> None:
    """The first entry's outgoing can be handed over explicitly."""
    control = DJControls()
    prefs: dict[str, tuple[str, str]] = _automix_blend_prefs(
        [{"queue_item_id": "q1", "genre": "Klezmer"}],
        control,
        previous={"genre": "Hip-Hop"},
    )
    assert prefs["q1"] == ("radio", "strict")


def test_blend_prefs_user_pin_unchanged() -> None:
    """An explicit user pin keeps today's exact pref values."""
    control = DJControls(transition_style="echo_out", beat_strictness="aggressive")
    prefs: dict[str, tuple[str, str]] = _automix_blend_prefs(
        [{"queue_item_id": "q1", "genre": "House"}], control
    )
    assert prefs["q1"] == ("echo_out", "aggressive")


def test_blend_prefs_skips_entries_without_ids() -> None:
    """Entries without a queue_item_id are skipped like before."""
    control = DJControls()
    prefs: dict[str, tuple[str, str]] = _automix_blend_prefs(
        [{"genre": "House"}, {"queue_item_id": "q1", "genre": "Pop"}], control
    )
    assert list(prefs) == ["q1"]
    assert prefs["q1"] == ("radio", "aggressive")


def test_pair_style_tolerates_none_outgoing() -> None:
    """outgoing=None must behave like the old single-sided read."""
    control = DJControls()
    resolved: dict[str, Any] = _pair_style(None, {"genre": "House"}, control)
    assert resolved["style"] == "full_blend"
    assert resolved["source"] == "genre"
