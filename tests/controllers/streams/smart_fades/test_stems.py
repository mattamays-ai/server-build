"""Tests for smart_fades/stems.py: PCM bridge, resampling, vocal ducking, masks."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from music_assistant.controllers.streams.smart_fades import stems
from music_assistant.controllers.streams.smart_fades.models import BlendPrefs
from music_assistant.controllers.streams.smart_fades.stems import (
    StemSeparator,
    duck_db_for_mask,
    duck_pcm_bytes,
    numpy_to_pcm_bytes,
    pcm_bytes_to_numpy,
    resample_linear,
)

RNG = np.random.default_rng(20250101)


# --- pcm_bytes_to_numpy / numpy_to_pcm_bytes ------------------------------------


def test_pcm16_decode_known_values() -> None:
    """16-bit LE interleaved PCM decodes to exactly v / 32768, deinterleaved."""
    pcm = np.array([1000, -2000, 32767, -32768, 0, 5], dtype="<i2").tobytes()
    out = pcm_bytes_to_numpy(pcm, 44100, 16, 2)
    assert out is not None
    assert out.dtype == np.float32
    assert out.shape == (2, 3)
    expected = (
        np.array([[1000, 32767, 0], [-2000, -32768, 5]], dtype=np.int64) / 32768.0
    ).astype(np.float32)
    # every v / 2**15 is exactly representable in float32
    np.testing.assert_array_equal(out, expected)


def test_pcm16_encode_known_bytes() -> None:
    """Encoding known floats yields the expected truncated-toward-zero int16 bytes."""
    wave = np.array([[0.5, -0.5, 0.0], [0.25, -0.25, 1.0]], dtype=np.float32)
    encoded = numpy_to_pcm_bytes(wave, 16, 2)
    assert encoded is not None
    np.testing.assert_array_equal(
        np.frombuffer(encoded, dtype="<i2"),
        np.array([16383, 8191, -16383, -8191, 0, 32767], dtype="<i2"),
    )


def test_pcm16_roundtrip_within_1_lsb() -> None:
    """bytes -> float -> bytes stays within one LSB over a large random sample."""
    values = RNG.integers(-32768, 32768, size=100000, dtype=np.int32)
    pcm = values.astype("<i2").tobytes()
    out = pcm_bytes_to_numpy(pcm, 44100, 16, 2)
    assert out is not None
    back = np.frombuffer(numpy_to_pcm_bytes(out, 16, 2), dtype="<i2").astype(np.int64)
    assert np.abs(back - values).max() <= 1


def test_pcm16_roundtrip_exact_for_zero() -> None:
    """An all-zero 16-bit buffer round-trips to byte-identical output."""
    pcm = np.zeros(4800, dtype="<i2").tobytes()
    out = pcm_bytes_to_numpy(pcm, 44100, 16, 2)
    assert out is not None
    assert numpy_to_pcm_bytes(out, 16, 2) == pcm


def test_pcm32_roundtrip_exact() -> None:
    """32-bit PCM round-trips to byte-identical output for safe-range values."""
    values = RNG.integers(-(1 << 23), 1 << 23, size=100000, dtype=np.int64)
    pcm = values.astype("<i4").tobytes()
    out = pcm_bytes_to_numpy(pcm, 44100, 32, 2)
    assert out is not None
    assert numpy_to_pcm_bytes(out, 32, 2) == pcm
    # and the decode of known bytes is exact
    known = np.array([100000, -200000, 4096, -4096], dtype="<i4").tobytes()
    decoded = pcm_bytes_to_numpy(known, 44100, 32, 2)
    assert decoded is not None
    np.testing.assert_array_equal(
        decoded,
        (np.array([[100000, 4096], [-200000, -4096]], dtype=np.int64) / 2147483648.0).astype(
            np.float32
        ),
    )


def test_pcm24_decode_known_triples() -> None:
    """24-bit packed LE triples decode to signed v / 2**23 with correct order."""
    pcm = bytes(
        [
            0x01,
            0x00,
            0x00,  # +1
            0xFF,
            0xFF,
            0xFF,  # -1
            0x00,
            0x00,
            0x80,  # -8388608 (full scale negative)
            0xFF,
            0xFF,
            0x7F,  # +8388607 (full scale positive)
        ]
    )
    out = pcm_bytes_to_numpy(pcm, 44100, 24, 2)
    assert out is not None
    assert out.dtype == np.float32
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(
        out,
        np.array([[1 / 8388608.0, -1.0], [-1 / 8388608.0, 8388607 / 8388608.0]], np.float32),
    )


def test_pcm_bridge_rejects_unsupported() -> None:
    """Non-stereo or unknown bit depths return None on both sides."""
    assert pcm_bytes_to_numpy(b"\x00" * 8, 44100, 16, 1) is None
    assert pcm_bytes_to_numpy(b"\x00" * 8, 44100, 8, 2) is None
    wave = np.zeros((2, 4), dtype=np.float32)
    assert numpy_to_pcm_bytes(wave, 16, 1) is None
    assert numpy_to_pcm_bytes(wave, 17, 2) is None


# --- resample_linear ------------------------------------------------------------


def test_resample_identity_when_rates_equal() -> None:
    wave = RNG.standard_normal((2, 4096)).astype(np.float32)
    out = resample_linear(wave, 48000, 48000)
    assert out is wave or np.array_equal(out, wave)


def test_resample_44100_to_48000_length() -> None:
    wave = RNG.standard_normal((2, 44100)).astype(np.float32)
    out = resample_linear(wave, 44100, 48000)
    assert out.shape == (2, 48000)
    assert out.dtype == np.float32


def test_resample_zero_stays_zero() -> None:
    n_out = max(1, int(round(1000 * 48000 / 44100)))
    out = resample_linear(np.zeros((2, 1000), dtype=np.float32), 44100, 48000)
    assert out.shape == (2, n_out)
    assert np.array_equal(out, np.zeros_like(out))


# --- duck_pcm_bytes degradation negative control ---------------------------------


def test_duck_pcm_bytes_identity_without_model() -> None:
    """With no model file present the duck must return the input bytes unchanged."""
    pcm = RNG.integers(-20000, 20000, size=4800).astype("<i2").tobytes()
    assert stems.available() is False  # environmental precondition of this control
    assert duck_pcm_bytes(pcm, 48000, 16, 2) == pcm


# --- duck_db_for_mask ------------------------------------------------------------


def test_duck_db_for_mask_synthetic() -> None:
    zeros = np.zeros((2, 512), dtype=np.float32)
    ones = np.ones((2, 512), dtype=np.float32)
    half = np.full((2, 512), 0.5, dtype=np.float32)
    assert duck_db_for_mask(zeros) == pytest.approx(0.0, abs=1e-12)
    assert duck_db_for_mask(ones) == pytest.approx(-6.0)
    assert duck_db_for_mask(half) == pytest.approx(-3.0)
    # depth scales linearly with the nominal depth
    assert duck_db_for_mask(ones, duck_db=-12.0) == pytest.approx(-12.0)


# --- BlendPrefs -------------------------------------------------------------------


def test_blendprefs_stems_default_false() -> None:
    prefs = BlendPrefs()
    assert prefs.stems is False


def test_blendprefs_is_frozen() -> None:
    prefs = BlendPrefs()
    with pytest.raises(dataclasses.FrozenInstanceError):
        prefs.stems = True  # type: ignore[misc]


# --- get_mask bounds with a stubbed model ------------------------------------------


def test_get_mask_shape_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a stubbed session/window the mask matches the rfft bin layout, 0..1."""
    monkeypatch.setattr(StemSeparator, "_get_session", lambda self: object())
    # synthetic "vocal estimate": the mix itself, scaled down by half
    monkeypatch.setattr(
        StemSeparator, "_run_window", lambda self, session, wave: (wave * 0.5).astype(np.float32)
    )
    separator = StemSeparator()
    mix = RNG.standard_normal((2, 4096)).astype(np.float32)
    mask = separator.get_mask(mix)
    assert mask is not None
    expected_bins = np.fft.rfft(mix, axis=1).shape[1]
    assert mask.shape == (2, expected_bins)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
