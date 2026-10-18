"""Tests for the derived per-transition transition_bars (0.0.28 lane C)."""

from __future__ import annotations

import re

from music_assistant.providers.smart_dj.engine import (
    ARC_CLIMB_MAX,
    ARC_RESET_DROP,
    DJControls,
)
from music_assistant.providers.smart_dj.provider import (
    RULE_TRANSITION_BARS_CAPPED,
    TRANSITION_BARS_REASON_CLIMB,
    TRANSITION_BARS_REASON_FAMILY,
    TRANSITION_BARS_REASON_GLOBAL,
    TRANSITION_BARS_REASON_OVERRIDE,
    TRANSITION_BARS_REASON_RESET,
    TRANSITION_BARS_REASON_VOCAL,
    _automix_durations,
    _pair_style,
    _transition_report,
    derive_transition_bars,
)


def _control(bars: int = 8, overrides: dict[str, int] | None = None) -> DJControls:
    """Build DJControls with just the fields the derivation reads."""
    return DJControls(transition_bars=bars, transition_bars_overrides=overrides or {})


def test_default_house_pair_keeps_global_bars() -> None:
    """Default 8 on a house pair sits inside the family candidates: no change."""
    bars, reason = derive_transition_bars({"genre": "House"}, {"genre": "Deep House"}, _control())
    assert bars == 8
    assert reason == TRANSITION_BARS_REASON_GLOBAL


def test_house_climb_gets_the_long_build() -> None:
    """A climb (energy up >= ARC_CLIMB_MAX) takes the family's long stop."""
    outgoing = {"genre": "House", "energy": 0.4}
    incoming = {"genre": "Techno", "energy": 0.4 + ARC_CLIMB_MAX}
    assert derive_transition_bars(outgoing, incoming, _control()) == (
        32,
        TRANSITION_BARS_REASON_CLIMB,
    )


def test_reset_gets_the_short_stop() -> None:
    """A reset drop (<= -ARC_RESET_DROP) takes the family's short stop."""
    outgoing = {"genre": "House", "energy": 0.8}
    incoming = {"genre": "House", "energy": 0.8 - ARC_RESET_DROP}
    assert derive_transition_bars(outgoing, incoming, _control(bars=16)) == (
        8,
        TRANSITION_BARS_REASON_RESET,
    )


def test_family_clamps_an_oversized_base() -> None:
    """A pop pair's phrase math derives a 32-bar base down to 8 bars."""
    assert derive_transition_bars({"genre": "Pop"}, {"genre": "Hip-Hop"}, _control(bars=32)) == (
        8,
        TRANSITION_BARS_REASON_FAMILY,
    )


def test_dnb_never_exceeds_its_verified_ceiling() -> None:
    """DnB tops out at 8 bars (the PAIR_BAR_CEILINGS verdict), climb or not."""
    outgoing = {"genre": "Drum and Bass", "energy": 0.3}
    incoming = {"genre": "Jungle", "energy": 0.9}
    assert derive_transition_bars(outgoing, incoming, _control(bars=8)) == (
        8,
        TRANSITION_BARS_REASON_CLIMB,
    )
    # a 16-bar user base is family-clamped to 8 first; the climb then holds
    assert derive_transition_bars(outgoing, incoming, _control(bars=16)) == (
        8,
        TRANSITION_BARS_REASON_CLIMB,
    )


def test_vocal_pair_stays_short_even_on_a_climb() -> None:
    """A KNOWN vocal-vs-vocal boundary always falls back to the short stop."""
    outgoing = {"genre": "House", "energy": 0.3, "instrumental": False}
    incoming = {"genre": "House", "energy": 0.9, "instrumental": False}
    assert derive_transition_bars(outgoing, incoming, _control()) == (
        8,
        TRANSITION_BARS_REASON_VOCAL,
    )


def test_instrumental_boundaries_are_not_vocal_shortened() -> None:
    """A known-instrumental boundary keeps the climb's long build."""
    outgoing = {"genre": "House", "energy": 0.3, "instrumental": True}
    incoming = {"genre": "House", "energy": 0.9, "instrumental": True}
    assert derive_transition_bars(outgoing, incoming, _control()) == (
        32,
        TRANSITION_BARS_REASON_CLIMB,
    )


def test_unknown_vocal_status_does_not_shorten() -> None:
    """Missing instrumental flags must not trigger the vocal shortening."""
    outgoing = {"genre": "Pop", "energy": 0.5}
    incoming = {"genre": "Hip-Hop", "energy": 0.55}
    assert derive_transition_bars(outgoing, incoming, _control(bars=8)) == (
        8,
        TRANSITION_BARS_REASON_GLOBAL,
    )


def test_cross_family_pair_takes_the_stricter_stop() -> None:
    """House -> pop spans families: the tightest (short, long) governs."""
    assert derive_transition_bars({"genre": "House"}, {"genre": "Pop"}, _control(bars=32)) == (
        8,
        TRANSITION_BARS_REASON_FAMILY,
    )


def test_unprofiled_genres_pass_the_global_through() -> None:
    """No family match: the user's global bars are kept verbatim."""
    for bars_in in (4, 8, 16, 32):
        assert derive_transition_bars(
            {"genre": "Klezmer"}, {"genre": "Polka"}, _control(bars=bars_in)
        ) == (bars_in, TRANSITION_BARS_REASON_GLOBAL)


def test_missing_energy_skips_the_arc() -> None:
    """No usable energies: family bounds apply, but no arc reason fires."""
    outgoing = {"genre": "House", "energy": "loud"}
    incoming = {"genre": "Techno"}
    assert derive_transition_bars(outgoing, incoming, _control(bars=8)) == (
        8,
        TRANSITION_BARS_REASON_GLOBAL,
    )


def test_a_legit_zero_energy_is_not_falsy_swallowed() -> None:
    """Energy 0.0 on one side still drives the arc (M-5 class regression)."""
    outgoing = {"genre": "House", "energy": 0.0}
    incoming = {"genre": "Techno", "energy": 0.0 + ARC_CLIMB_MAX}
    assert derive_transition_bars(outgoing, incoming, _control()) == (
        32,
        TRANSITION_BARS_REASON_CLIMB,
    )


def test_derivation_is_deterministic() -> None:
    """Same inputs, same (bars, reason) - the decision is pure."""
    outgoing = {"genre": "House", "energy": 0.2, "instrumental": False}
    incoming = {"genre": "Techno", "energy": 0.75, "instrumental": False}
    expected = derive_transition_bars(outgoing, incoming, _control(bars=16))
    assert expected == derive_transition_bars(dict(outgoing), dict(incoming), _control(bars=16))


def test_pair_style_reports_reason_and_capped_rule() -> None:
    """_pair_style exposes bars_reason and the capped rule on a family clamp."""
    resolved = _pair_style({"genre": "Pop"}, {"genre": "Hip-Hop"}, _control(bars=32))
    assert resolved["bars"] == 8
    assert resolved["bars_reason"] == TRANSITION_BARS_REASON_FAMILY
    assert RULE_TRANSITION_BARS_CAPPED in resolved["rules"]


def test_pair_style_family_rule_not_fired_when_base_inside_family() -> None:
    """Default 8 on a house pair: no family clamp, no capped rule."""
    resolved = _pair_style({"genre": "House"}, {"genre": "Techno"}, _control())
    assert resolved["bars"] == 8
    assert resolved["bars_reason"] == TRANSITION_BARS_REASON_GLOBAL
    assert resolved["rules"] == []


def test_user_override_wins_in_the_report() -> None:
    """A per-track transition_bars override beats the derivation (0 = hard cut)."""
    control = _control(overrides={"q1": 0})
    ordered = [
        {"name": "A", "queue_item_id": "q0", "genre": "House"},
        {"name": "B", "queue_item_id": "q1", "genre": "House"},
    ]
    notes = _transition_report(ordered, control)
    assert notes[0]["transition_bars"] == 0
    assert notes[0]["transition_bars_reason"] == TRANSITION_BARS_REASON_OVERRIDE


def test_report_carries_derived_bars_and_reason() -> None:
    """Report rows expose the derived value + reason when no override exists."""
    ordered = [
        {"name": "A", "queue_item_id": "q0", "genre": "House"},
        {"name": "B", "queue_item_id": "q1", "genre": "Pop"},
    ]
    notes = _transition_report(ordered, _control(bars=32))
    assert notes[0]["transition_bars"] == 8
    assert notes[0]["transition_bars_reason"] == TRANSITION_BARS_REASON_FAMILY


def test_reason_ids_are_stable_snake_case() -> None:
    """Every emitted reason id is snake_case (the FE reads it verbatim)."""
    for reason in (
        TRANSITION_BARS_REASON_GLOBAL,
        TRANSITION_BARS_REASON_FAMILY,
        TRANSITION_BARS_REASON_CLIMB,
        TRANSITION_BARS_REASON_RESET,
        TRANSITION_BARS_REASON_VOCAL,
        TRANSITION_BARS_REASON_OVERRIDE,
    ):
        assert re.fullmatch(r"[a-z][a-z0-9_]*", reason), reason


def test_automix_durations_use_derived_bars() -> None:
    """Without overrides the pin uses the derived count (pop pair: 8 bars)."""
    tracks = [
        {"queue_item_id": "q0", "genre": "Pop", "analysis": {"bpm": 120, "energy": 0.5}},
        {"queue_item_id": "q1", "genre": "Hip-Hop", "analysis": {"bpm": 90, "energy": 0.55}},
    ]
    durations = _automix_durations(tracks[1:], _control(bars=32), previous=tracks[0])
    # 8 bars at 90 bpm, 4 beats per bar -> 8 * 4 * 60 / 90 = 21.33
    assert durations == {"q1": 21.33}


def test_automix_override_still_wins() -> None:
    """A per-track override beats the derived count in the pins."""
    tracks = [{"queue_item_id": "q1", "genre": "Pop", "analysis": {"bpm": 120, "energy": 0.5}}]
    durations = _automix_durations(tracks, _control(overrides={"q1": 16}))
    assert durations == {"q1": 32.0}
