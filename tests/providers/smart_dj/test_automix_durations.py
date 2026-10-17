"""Tests for Smart DJ AutoMix custom mix length derivation."""

from typing import Any

from music_assistant.providers.smart_dj.engine import DJControls
from music_assistant.providers.smart_dj.provider import AUTOMIX_MAX_CROSSFADE, _automix_durations


def _control(bars: int = 8, overrides: dict[str, int] | None = None) -> DJControls:
    """Build DJControls with just the fields _automix_durations reads."""
    return DJControls(transition_bars=bars, transition_bars_overrides=overrides or {})


def test_duration_from_global_bars() -> None:
    """8 bars at 120 bpm with 4 beats per bar is exactly 16 seconds."""
    tracks = [{"queue_item_id": "q1", "analysis": {"bpm": 120, "beats_per_bar": 4}}]
    assert _automix_durations(tracks, _control()) == {"q1": 16.0}


def test_override_wins_over_global() -> None:
    """A per-track bars override replaces the global transition_bars."""
    tracks = [{"queue_item_id": "q1", "analysis": {"bpm": 120, "beats_per_bar": 4}}]
    durations = _automix_durations(tracks, _control(bars=32, overrides={"q1": 4}))
    assert durations == {"q1": 8.0}


def test_track_without_bpm_gets_no_pin() -> None:
    """Tracks without usable bpm keep the engine's default length (no pin)."""
    tracks: list[dict[str, Any]] = [
        {"queue_item_id": "q1", "analysis": None},
        {"queue_item_id": "q2", "analysis": {"bpm": "nonsense", "beats_per_bar": 4}},
        {"queue_item_id": "q3", "analysis": {"bpm": 0, "beats_per_bar": 4}},
    ]
    assert _automix_durations(tracks, _control()) == {}


def test_slow_32_bar_pin_is_capped() -> None:
    """32 bars at 60 bpm would be 128s; the pin is capped for sane PCM hold-back."""
    tracks = [{"queue_item_id": "q1", "analysis": {"bpm": 60, "beats_per_bar": 4}}]
    durations = _automix_durations(tracks, _control(bars=32))
    assert durations == {"q1": AUTOMIX_MAX_CROSSFADE}


def test_bad_beats_per_bar_falls_back_to_four() -> None:
    """A non-numeric beats_per_bar falls back to 4 beats per bar."""
    tracks = [{"queue_item_id": "q1", "analysis": {"bpm": 120, "beats_per_bar": "4/4"}}]
    assert _automix_durations(tracks, _control()) == {"q1": 16.0}


def test_flat_bpm_fallback() -> None:
    """Track dicts without a nested analysis can still pin from a flat bpm."""
    tracks = [{"queue_item_id": "q1", "bpm": 100, "beats_per_bar": 4}]
    assert _automix_durations(tracks, _control()) == {"q1": 19.2}
