"""
Smart Fades - vocal/instrumental separation for stem-aware ducking.

Splits a stereo mix into a vocal estimate so the outgoing track's vocal can be
ducked independently of its instrumental bed. The separation model (MDX-Net
``Kim_Vocal_2``) is a *soft* dependency: ``onnxruntime`` and the model file are
both optional, and this module imports cleanly without either. Every entry
point degrades to ``None`` when the stack is absent, so the caller keeps its
exact current behavior and no code path acquires a hard requirement.

The model file is fetched once on first use and verified by SHA-256 before it
is ever loaded; masks are cached per ``(item_id, provider, sample_rate)``
because separation is far more expensive than the transition it serves.

Inference notes (proven against the prototype in 0.0.25 batch 5 dev):
the graph is MDX-Net with n_fft 7680 / hop 1024 / dim_f 3072 / dim_t 256;
"denoising" means the mean of a +input and -input pass; the Hann window is
the periodic (0.5 - 0.5*cos) variant and torch center=True equals a constant
zero-pad of TRIM on both sides. STFT/ISTFT round-trip against librosa 1.0.0
matches to 6.7e-08 relative error.
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from importlib import import_module
from types import ModuleType

import numpy as np

LOGGER = logging.getLogger(__name__)

# MDX-Net Kim_Vocal_2 graph geometry. These are properties of the shipped
# model, not tunables: changing any of them changes what the graph computes.
N_FFT = 7680
HOP = 1024
DIM_F = 3072
DIM_T = 256
COMPENSATE = 1.009
SAMPLE_RATE = 44100
N_BINS = N_FFT // 2 + 1
CHUNK_SIZE = HOP * (DIM_T - 1)
TRIM = N_FFT // 2
GEN_SIZE = CHUNK_SIZE - 2 * TRIM
WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)

# Where the model lands when a user enables stems. Absent = feature off.
MODEL_DIR = Path("/data/smartdj/models")
MODEL_FILE = MODEL_DIR / "Kim_Vocal_2.onnx"
MODEL_URL = "https://huggingface.co/seanghay/uvr_models/resolve/main/Kim_Vocal_2.onnx"
MODEL_SHA256 = "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b"
MODEL_SIZE_BYTES = 66_759_214  # 66.8 MB measured; earlier "30-60MB" estimate was low

# Duck depth applied to the outgoing vocal during a fully vocal overlap.
DEFAULT_DUCK_DB = -6.0
# Bounded mask cache: separation is expensive, but a player could crossfade
# through many tracks per session, so never let this grow without a cap.
MASK_CACHE_MAX = 8


def _onnxruntime() -> ModuleType | None:
    """Import onnxruntime lazily - it is an optional dependency."""
    try:
        return import_module("onnxruntime")
    except ImportError:
        return None


def available() -> bool:
    """Return True when both the runtime and the model file are present."""
    if _onnxruntime() is None:
        return False
    return MODEL_FILE.is_file()


def ensure_model() -> bool:
    """
    Fetch the separation model if absent, verifying SHA-256 before use.

    Runs blocking network + disk IO, so callers must hop to a thread. Returns
    True when a verified model file is in place afterwards.
    """
    if _sha256_of(MODEL_FILE) == MODEL_SHA256:
        return True
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        staging = MODEL_DIR / "Kim_Vocal_2.onnx.part"
        with urllib.request.urlopen(MODEL_URL, timeout=120) as resp, staging.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        if _sha256_of(staging) != MODEL_SHA256:
            staging.unlink(missing_ok=True)
            LOGGER.warning("Stem model download failed SHA-256 verification; discarding")
            return False
        staging.replace(MODEL_FILE)
        return True
    except Exception as err:  # noqa: BLE001 - any fetch failure just means "off"
        LOGGER.warning("Stem model download failed: %s", err)
        return False


def _sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, or "" when it does not exist."""
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class StemCacheEntry:
    """A separated vocal mask plus the identity it was computed for."""

    item_id: str
    provider: str
    sample_rate: int
    mask: np.ndarray
    created_at: float

    def matches(self, item_id: str, provider: str, sample_rate: int) -> bool:
        """Return True when this entry is valid for the requested identity."""
        return (
            self.item_id == item_id
            and self.provider == provider
            and self.sample_rate == sample_rate
        )


class StemSeparator:
    """Loads the MDX-Net graph once and separates stereo mixes into vocals."""

    def __init__(self, logger: logging.Logger = LOGGER) -> None:
        """Initialize the separator without touching onnxruntime."""
        self.logger = logger
        self._session: object | None = None
        self._cache: dict[tuple[str, str, int], StemCacheEntry] = {}
        self._model_verified = False

    def separate(self, mix: np.ndarray) -> np.ndarray | None:
        """
        Separate a stereo mix into an estimated vocal waveform.

        :param mix: float32 array shaped (2, samples) at ``SAMPLE_RATE``.
        :return: vocals shaped (2, samples), or None when stems are unavailable.
        """
        session = self._get_session()
        if session is None:
            return None
        n_sample = mix.shape[1]
        pad = GEN_SIZE - n_sample % GEN_SIZE
        mix_p = np.concatenate(
            (np.zeros((2, TRIM), np.float32), mix, np.zeros((2, pad + TRIM), np.float32)),
            axis=1,
        )
        windows = []
        i = 0
        while i < n_sample + pad:
            windows.append(mix_p[:, i : i + CHUNK_SIZE])
            i += GEN_SIZE
        tars = [self._run_window(session, window) for window in windows]
        target = np.concatenate(tars, axis=1)[:, : n_sample + pad]
        return target[:, :n_sample]

    def get_mask(self, mix: np.ndarray) -> np.ndarray | None:
        """
        Return a bounded Wiener mask (0..1) for the mix's vocal content.

        :param mix: float32 array shaped (2, samples) at ``SAMPLE_RATE``.
        :return: mask shaped (2, bins) as float32 (rfft bin layout, same length
            as ``np.fft.rfft(mix, axis=1)``), or None when stems are unavailable.
        """
        vocals = self.separate(mix)
        if vocals is None:
            return None
        voc_spec = np.abs(np.fft.rfft(vocals, axis=1))
        instr_spec = np.abs(np.fft.rfft(mix - vocals, axis=1))
        v2, i2 = voc_spec**2, instr_spec**2
        mask: np.ndarray = v2 / np.maximum(v2 + i2, 1e-12)
        bounded: np.ndarray = np.clip(mask, 0.0, 1.0).astype(np.float32)
        return bounded

    def cached_mask(
        self, item_id: str, provider: str, sample_rate: int, mix: np.ndarray
    ) -> np.ndarray | None:
        """
        Return a cached mask for the given track identity, computing if needed.

        :param item_id: the media item identity the mask belongs to.
        :param provider: the provider that served the track.
        :param sample_rate: the sample rate the mask was computed at.
        :param mix: float32 array shaped (2, samples) at ``SAMPLE_RATE``.
        :return: mask shaped (2, bins), or None when stems are unavailable.
        """
        key = (item_id, provider, sample_rate)
        entry = self._cache.get(key)
        if entry is not None and entry.matches(item_id, provider, sample_rate):
            return entry.mask
        mask = self.get_mask(mix)
        if mask is None:
            return None
        self._cache[key] = StemCacheEntry(
            item_id=item_id,
            provider=provider,
            sample_rate=sample_rate,
            mask=mask,
            created_at=time.time(),
        )
        while len(self._cache) > MASK_CACHE_MAX:
            oldest = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest]
        return mask



    def _verify_model(self) -> bool:
        """Check the model file is the expected artifact before loading it."""
        if self._model_verified:
            return True
        if _sha256_of(MODEL_FILE) != MODEL_SHA256:
            self.logger.warning("Stem model file missing or failed SHA-256 verification")
            return False
        self._model_verified = True
        return True

    def _get_session(self) -> object | None:
        """Build (once) and return an inference session, or None if unavailable."""
        if self._session is not None:
            return self._session
        ort = _onnxruntime()
        if ort is None:
            return None
        if not self._verify_model():
            return None
        options = ort.SessionOptions()
        options.log_severity_level = 3
        # The arena grows to the graph's activation peak and never releases it,
        # which is fatal on memory-constrained hosts: keep it off.
        options.enable_cpu_mem_arena = False
        try:
            self._session = ort.InferenceSession(
                str(MODEL_FILE), options, providers=["CPUExecutionProvider"]
            )
        except Exception as err:  # noqa: BLE001 - any load failure means "off"
            self.logger.warning("Stem model failed to load: %s", err)
            return None
        return self._session

    @staticmethod
    def _stft(wave: np.ndarray) -> np.ndarray:
        """Compute the 4-channel MDX spectrogram from a stereo window."""
        padded = np.pad(wave, ((0, 0), (0, 0), (TRIM, TRIM)))
        frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT, axis=2)[
            :, :, ::HOP, :
        ]
        spec = np.fft.rfft(frames * WINDOW.astype(np.float32), n=N_FFT, axis=-1).transpose(
            0, 1, 3, 2
        )
        out = np.empty((wave.shape[0], 4, N_BINS, spec.shape[-1]), dtype=np.float32)
        out[:, 0], out[:, 1] = spec[:, 0].real, spec[:, 0].imag
        out[:, 2], out[:, 3] = spec[:, 1].real, spec[:, 1].imag
        return out[:, :, :DIM_F, :]

    @staticmethod
    def _istft(spec_c: np.ndarray, chunk_size: int) -> np.ndarray:
        """Invert a complex spectrogram back to a stereo waveform."""
        n_frames = spec_c.shape[-1]
        y = np.zeros((spec_c.shape[0], 2, (n_frames - 1) * HOP + N_FFT), dtype=np.float64)
        wsum = np.zeros((n_frames - 1) * HOP + N_FFT, dtype=np.float64)
        frames = np.fft.irfft(spec_c, n=N_FFT, axis=-2).transpose(0, 1, 3, 2)
        w2 = WINDOW.astype(np.float64) ** 2
        for t in range(n_frames):
            y[:, :, t * HOP : t * HOP + N_FFT] += frames[:, :, t, :] * WINDOW
            wsum[t * HOP : t * HOP + N_FFT] += w2
        y /= np.maximum(wsum, 1e-11)
        return y[:, :, TRIM : TRIM + chunk_size]

    def _run_window(self, session: object, wave: np.ndarray) -> np.ndarray:
        """Run the denoising MDX pass on one window and return its vocal waveform."""
        spek = self._stft(wave[None].astype(np.float32))
        # run() returns a list of tensors and the batch dim survives: [0][0]
        pos = session.run(None, {"input": spek})[0][0]  # type: ignore[attr-defined]
        neg = session.run(None, {"input": -spek})[0][0]  # type: ignore[attr-defined]
        spec_pred = (pos - neg) * 0.5 * COMPENSATE
        full = np.zeros((2, N_BINS, DIM_T), dtype=np.complex128)
        for ch in range(2):
            full[ch, :DIM_F] = spec_pred[2 * ch] + 1j * spec_pred[2 * ch + 1]
        target: np.ndarray = self._istft(full[None], CHUNK_SIZE)[0]
        return target[:, TRIM:-TRIM]




def duck_db_for_mask(mask: np.ndarray, duck_db: float = DEFAULT_DUCK_DB) -> float:
    """
    Scale a fixed duck depth by how much vocal energy the mask actually carries.

    A quiet or instrumental-only stretch should not be ducked as hard as a
    vocal-forward one, so the nominal depth is weighted by mean mask energy.

    :param mask: bounded mask shaped (2, bins) from :meth:`StemSeparator.get_mask`.
    :param duck_db: the nominal duck depth for a fully vocal passage.
    :return: the effective duck depth in dB (always <= 0).
    """
    energy = float(np.clip(mask.mean(), 0.0, 1.0))
    return duck_db * energy


# --- PCM bridge -----------------------------------------------------------------
# These helpers keep fades.py thin: the fade layer hands over raw PCM bytes and
# format primitives and gets PCM bytes back. All numpy DSP lives here.

_DTYPES = {16: "<i2", 24: None, 32: "<i4"}


def pcm_bytes_to_numpy(
    pcm: bytes, sample_rate: int, bit_depth: int, channels: int
) -> np.ndarray | None:
    """
    Decode little-endian interleaved PCM bytes to float32 shaped (channels, samples).

    :param pcm: the raw PCM bytes.
    :param sample_rate: PCM sample rate (checked, unused in the decode itself).
    :param bit_depth: 16, 24 or 32; anything else returns None.
    :param channels: only stereo (2) is supported; anything else returns None.
    :return: float32 array in the -1..1 range, or None when unsupported.
    """
    if channels != 2 or bit_depth not in _DTYPES:
        return None
    if bit_depth == 16:
        data = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif bit_depth == 32:
        data = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    else:  # 24-bit packed little-endian
        raw = np.frombuffer(pcm, dtype=np.uint8)
        usable = len(raw) - len(raw) % 3
        if usable == 0:
            return None
        triples = raw[:usable].reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values >= 1 << 23, values - (1 << 24), values)
        data = values.astype(np.float32) / float(1 << 23)
    trimmed = data.size - data.size % channels
    if trimmed == 0:
        return None
    data = data[:trimmed]
    return np.stack([data[ch::channels] for ch in range(channels)])


def numpy_to_pcm_bytes(
    wave: np.ndarray, bit_depth: int, channels: int
) -> bytes | None:
    """
    Encode float32 shaped (channels, samples) back to little-endian interleaved PCM.

    :param wave: float32 array in the -1..1 range.
    :param bit_depth: 16, 24 or 32; anything else returns None.
    :param channels: only stereo (2) is supported; anything else returns None.
    :return: the PCM bytes, or None when unsupported.
    """
    if channels != 2 or bit_depth not in _DTYPES or wave.ndim != 2:
        return None
    clipped = np.clip(wave, -1.0, 1.0)
    if bit_depth == 16:
        # native-endian int16: the add-on targets linux/amd64 + arm64, both LE
        out: np.ndarray = (clipped * 32767.0).astype(np.int16).T.reshape(-1)
        return out.tobytes()
    if bit_depth == 32:
        out = (clipped * 2147483647.0).astype(np.int32).T.reshape(-1)
        return out.tobytes()
    values = np.rint(clipped * float(1 << 23)).astype(np.int32)
    values = np.clip(values, -(1 << 23), (1 << 23) - 1)
    u = (values.T.reshape(-1)).astype(np.uint32)
    packed = np.empty((u.size, 3), dtype=np.uint8)
    packed[:, 0] = u & 0xFF
    packed[:, 1] = (u >> 8) & 0xFF
    packed[:, 2] = (u >> 16) & 0xFF
    return packed.tobytes()


def resample_linear(wave: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """
    Linearly resample along the sample axis of a (channels, samples) array.

    Good enough for mask computation and for the (small) difference signal the
    duck subtracts; the main signal never passes through the resampler.
    """
    if rate_in == rate_out:
        return wave.astype(np.float32, copy=False)
    n_in = wave.shape[1]
    n_out = max(1, int(round(n_in * rate_out / rate_in)))
    positions = np.arange(n_out) * (n_in / n_out)
    lo = np.floor(positions).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = (positions - lo).astype(np.float32)
    res: np.ndarray = (wave[:, lo] * (1.0 - frac)) + (wave[:, hi] * frac)
    return res.astype(np.float32)


def duck_pcm_bytes(
    pcm: bytes,
    sample_rate: int,
    bit_depth: int,
    channels: int,
    duck_db: float = DEFAULT_DUCK_DB,
    separator: StemSeparator | None = None,
) -> bytes:
    """
    Return the PCM with its vocal content suppressed by ``duck_db``.

    Identity-preserving: when stems are unavailable (no runtime, no model,
    unsupported format) the input bytes come back unchanged, so the caller
    needs no fallback branch. The overlap audio passes through the model at
    ``SAMPLE_RATE``; only the removed vocal signal is resampled back to the
    stream rate, keeping resampler artifacts out of the untouched bed.

    Runs CPU-bound model inference (roughly 2.3x the overlap duration on 8
    cores) - callers must hop to a thread and overlap it with playback.
    """
    if separator is None:
        separator = _SHARED_SEPARATOR
    try:
        mix = pcm_bytes_to_numpy(pcm, sample_rate, bit_depth, channels)
        if mix is None or mix.shape[1] < HOP:
            return pcm
        mix44 = resample_linear(mix, sample_rate, SAMPLE_RATE)
        # content-hash cache key (with the stream rate: identical bytes at
        # different rates resample to different masks, so they must not share)
        key = hashlib.sha1(pcm).hexdigest()[:16] + f"@{sample_rate}"  # noqa: S324
        mask = separator.cached_mask(key, "pcm", SAMPLE_RATE, mix44)
        if mask is None:
            return pcm
        spec = np.fft.rfft(mix44, axis=1)
        bins = min(spec.shape[1], mask.shape[1])
        vocal_gain = 1.0 - mask[:, :bins] * (1.0 - 10.0 ** (duck_db / 20.0))
        ducked44 = np.fft.irfft(spec[:, :bins] * vocal_gain, n=mix44.shape[1], axis=1)
        delta = resample_linear(
            (mix44 - ducked44).astype(np.float32), SAMPLE_RATE, sample_rate
        )
        if delta.shape[1] < mix.shape[1]:  # pad the tail after rounding
            delta = np.pad(delta, ((0, 0), (0, mix.shape[1] - delta.shape[1])))
        out = mix - delta[:, : mix.shape[1]]
        encoded = numpy_to_pcm_bytes(out.astype(np.float32), bit_depth, channels)
    except Exception:  # noqa: BLE001 - a duck must never break a transition
        LOGGER.exception("Stem duck failed; falling back to the unducked overlap")
        return pcm
    if encoded is None or len(encoded) != len(pcm):
        return pcm
    return encoded


_SHARED_SEPARATOR = StemSeparator()
