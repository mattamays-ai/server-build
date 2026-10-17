"""Tests for the radio (sequential handover) style and the boundary gain match."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.smart_fades.filters import (
    GainMatchFilter,
    SequentialCrossfadeFilter,
    StreamingCrossfadeFilter,
)
from music_assistant.controllers.streams.smart_fades.models import (
    Deck,
    EqPlan,
    PlanMetrics,
    TransitionPlan,
    TransitionStyle,
    TransitionTier,
)
from music_assistant.controllers.streams.smart_fades.planner.assembly import PlanAssembler
from music_assistant.controllers.streams.smart_fades.planner.candidates import (
    Candidate,
    RadioGenerator,
    default_generators,
)
from music_assistant.controllers.streams.smart_fades.planner.context import TransitionContext
from music_assistant.controllers.streams.smart_fades.renderer import TransitionRenderer
from music_assistant.models.audio_analysis import AudioAnalysisData

LOGGER = logging.getLogger(__name__)

PCM = AudioFormat(content_type=ContentType.PCM_S16LE, sample_rate=44100, bit_depth=16, channels=2)


def _seconds(n: float) -> int:
    """Return a head buffer of n seconds in PCM samples."""
    return int(n * 44100 * 2 * 2)  # stereo, 16-bit


def _analysis(
    bpm: float = 120.0,
    duration: float = 240.0,
    tail_quiet_from: float | None = None,
    head_quiet_until: float | None = None,
) -> AudioAnalysisData:
    """Build an analysis row with optional quiet windows in seconds (tail/head)."""
    interval = 60.0 / bpm
    beats = np.arange(0.0, duration, interval, dtype=np.float32)
    n_bins = 1800
    bin_s = duration / n_bins
    rms = np.full(n_bins, 0.5, dtype=np.float32)
    if tail_quiet_from is not None:
        rms[int(tail_quiet_from / bin_s) :] = 0.01
    if head_quiet_until is not None:
        rms[: int(head_quiet_until / bin_s)] = 0.01
    return AudioAnalysisData(
        duration=duration,
        bpm=bpm,
        beats=beats.tolist(),
        downbeats=beats[::4].tolist(),
        rms_energy=rms.tolist(),
        key="A",
        mode="minor",
    )


def _ctx(**overrides: Any) -> TransitionContext:
    """Build a minimal context; only the fields under test vary."""
    out_analysis = overrides.pop("out_analysis", _analysis())
    in_analysis = overrides.pop("in_analysis", _analysis())

    def _deck(analysis: AudioAnalysisData) -> Deck:
        return Deck(
            analysis=analysis,
            bpm=analysis.bpm or 120.0,
            beats=np.asarray(analysis.beats, dtype=np.float32),
            downbeats=np.asarray(analysis.downbeats, dtype=np.float32),
            beats_per_bar=analysis.beats_per_bar or 4,
        )

    base: dict[str, Any] = {
        "outgoing": _deck(out_analysis),
        "incoming": _deck(in_analysis),
        "outgoing_profile": None,
        "incoming_profile": None,
        "buffer_duration": 45.0,
        "buffer_offset": 0.0,
        "audio_end": 45.0,
        "default_anchor": 45.0,
        "mix_out_anchor": None,
        "kick_anchor": None,
        "fade_onset": None,
        "coda_zone": None,
        "tier": TransitionTier.FULL_BLEND,
        "cross_meter": False,
        "bpm_diff_percent": 0.0,
        "vocal_out_placement": None,
        "vocal_in_placement": None,
        "vocal_out_scoring": None,
        "vocal_in_scoring": None,
        "natural_entry": 0.0,
        "protective_downbeats": (),
    }
    base.update(overrides)
    return TransitionContext(**base)


def _plan(style: TransitionStyle = TransitionStyle.BLEND, **overrides: Any) -> TransitionPlan:
    """Build a minimal plan in the style under test (default: no tempo ramp)."""
    return TransitionPlan(
        tier=TransitionTier.FULL_BLEND,
        fade_out_window=40.0,
        crossfade_duration=8.0,
        style=style,
        **overrides,
    )


# --- radio generator ---


def test_radio_generator_is_silent_without_a_pin() -> None:
    """No style pinned: the radio generator emits nothing."""
    assert list(RadioGenerator().generate(_ctx())) == []


def test_radio_generator_emits_specs_when_pinned() -> None:
    """The radio pin emits specs carrying the RADIO style at the default anchor."""
    specs = list(RadioGenerator().generate(_ctx(style_override=TransitionStyle.RADIO)))
    assert specs, "a pinned radio style must emit candidates"
    for spec in specs:
        assert spec.style is TransitionStyle.RADIO
        assert spec.anchor_s is None
        assert spec.source == "radio"


def test_radio_generator_leads_the_default_registry() -> None:
    """The registry leads with the radio generator so a pin is never out-competed early."""
    assert isinstance(default_generators()[0], RadioGenerator)


# --- radio rendering ---


def test_renderer_uses_the_sequential_filter_for_radio() -> None:
    """A radio plan renders through the sequential filter, never the summing crossfade."""
    filters, _ = TransitionRenderer(LOGGER).render(_plan(TransitionStyle.RADIO), PCM, _seconds(45))
    assert any(isinstance(f, SequentialCrossfadeFilter) for f in filters)
    assert not any(isinstance(f, StreamingCrossfadeFilter) for f in filters)


def test_sequential_filter_never_sums_overlap() -> None:
    """The outgoing stops at the midpoint before the incoming starts; amix sums disjoint halves."""
    f = SequentialCrossfadeFilter(LOGGER, 44100 * 8, pre_crossfade_samples=44100)
    chains = f.apply("fadein_in", "fadeout_in")
    # outgoing: fades over the window's first half, then silence; trimmed at the
    # window end (pre + ns) so both streams end together for the summing amix
    assert f"afade=t=out:start_sample=44100:nb_samples={44100 * 4}" in chains[0]
    assert f"atrim=end_sample={44100 + 44100 * 8}" in chains[0]
    # incoming: delayed to pre + half, fading in over the second half
    assert f"adelay={44100 + 44100 * 4}S" in chains[1]
    assert f"afade=t=in:start_sample=0:nb_samples={44100 * 4}" in chains[1]


# --- radio assembly: neutral EQ, no gain match ---


def test_assembler_gives_radio_a_neutral_eq_and_no_gain_match() -> None:
    """Radio sums nothing: no EQ handover to stage and no boundary seam to match."""
    ctx = _ctx(style_override=TransitionStyle.RADIO)
    plan = _plan(TransitionStyle.RADIO)
    candidate = Candidate(
        spec=None,  # type: ignore[arg-type]  # finalize only reads .plan/.metrics
        plan=plan,
        metrics=PlanMetrics(),
        ideal_bars=8,
    )
    finalized = PlanAssembler(ctx, LOGGER).finalize(candidate)
    assert finalized.eq_plan == EqPlan.neutral(swap_at=0.6 * plan.crossfade_duration)
    assert finalized.gain_match_db is None


def test_assembler_keeps_gain_match_off_a_radio_plan() -> None:
    """The gain-match measurement explicitly refuses a radio plan."""
    ctx = _ctx(style_override=TransitionStyle.RADIO)
    assert PlanAssembler(ctx, LOGGER)._gain_match_db(_plan(TransitionStyle.RADIO)) is None


# --- gain match measurement ---


def test_gain_match_measures_a_loud_incoming_track() -> None:
    """A much louder incoming track earns a negative (pull-down) correction."""
    ctx = _ctx(
        out_analysis=_analysis(tail_quiet_from=32.0),  # the plan's fade window is 32-40s
        in_analysis=_analysis(),
    )
    db = PlanAssembler(ctx, LOGGER)._gain_match_db(_plan())
    assert db is not None
    assert db < 0, "a louder incoming deck must be pulled DOWN"


def test_gain_match_measures_a_quiet_incoming_track() -> None:
    """A much quieter incoming track earns a positive (lift) correction."""
    ctx = _ctx(out_analysis=_analysis(), in_analysis=_analysis(head_quiet_until=8.0))
    db = PlanAssembler(ctx, LOGGER)._gain_match_db(_plan())
    assert db is not None
    assert db > 0, "a quieter incoming deck must be lifted"


def test_gain_match_abstains_within_a_db() -> None:
    """Two equally loud decks: no correction (the boundary is already matched)."""
    ctx = _ctx()
    assert PlanAssembler(ctx, LOGGER)._gain_match_db(_plan()) is None


def test_gain_match_abstains_without_rms() -> None:
    """No analysis on either deck: abstain rather than guess."""
    out, inc = _analysis(), _analysis()
    out.rms_energy = None
    inc.rms_energy = None
    ctx = _ctx(out_analysis=out, in_analysis=inc)
    assert PlanAssembler(ctx, LOGGER)._gain_match_db(_plan()) is None


def test_gain_match_correction_is_clamped() -> None:
    """A wild loudness step is clamped, not faithfully corrected."""
    out_rms = np.full(1800, 0.5, dtype=np.float32)
    in_rms = np.full(1800, 0.0001, dtype=np.float32)
    out, inc = _analysis(), _analysis()
    out.rms_energy = out_rms.tolist()
    inc.rms_energy = in_rms.tolist()
    ctx = _ctx(out_analysis=out, in_analysis=inc)
    db = PlanAssembler(ctx, LOGGER)._gain_match_db(_plan())
    assert db is not None
    assert db <= 6.0


def test_gain_match_filter_envelope_shape() -> None:
    """The correction starts at full strength and returns to unity after the ramp."""
    f = GainMatchFilter(LOGGER, -2.0, overlap_seconds=8.0, ramp_seconds=2.0)
    chains = f.apply("fadein_in", "fadeout_in")
    # 10^(-2/20) = 0.7943... at the first sample; the envelope reaches 1.0 at t=10
    assert "0.79433" in chains[1], chains[1]
    assert "(10.000-t)/2.000" in chains[1], chains[1]
    lin = 10.0 ** (-2.0 / 20.0)
    assert math.isclose(lin, 0.7943282347242815, rel_tol=1e-6)


def test_gain_match_filter_is_wired_before_the_crossfade() -> None:
    """A plan with a correction renders the gain match ahead of the summing crossfade."""
    plan = _plan(gain_match_db=-2.0)
    filters, _ = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(45))
    kinds = [type(f) for f in filters]
    assert GainMatchFilter in kinds
    assert kinds.index(GainMatchFilter) < kinds.index(StreamingCrossfadeFilter)
