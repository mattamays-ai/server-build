"""
Derived-only quick-pull index for the Smart DJ provider (0.0.27 spec Item 3).

A tiny SQLite index in the Smart DJ provider's OWN storage dir
(``Path(mass.storage_path) / "smart_dj" / quick_index.sqlite3``) that carries
ONE derived row per analyzed track: bpm (plus the 80-160 normalized window and
its 2:1 half/double aliases), a Camelot key bucket, energy band/tier, mood
quadrant, vocal density tier, and era/year metadata. It deliberately stores NO
raw analysis arrays (beats, envelopes, embeddings) - those stay in
``audio_analysis.db``, which is version-checked and shared with other audio
analysis providers and must never gain tables (spec 3.1).

Rows are upserted at the moment an analysis is stored (the silent analyzer's
success point) and backfilled from ``iter_merged_audio_analysis_rows`` on
provider load, so ``smart_dj/pull_candidates`` and the planner's candidate-pool
build never need to deserialize the payload blobs.

Every value here is DERIVED: scalars from the already-stored
``AudioAnalysisData`` (models/audio_analysis.py) plus mood/release_date/year
from Music Assistant's own metadata (music_assistant_models metadata.py:191 and
:207) - zero new analyzer quota.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time as time_module
from contextlib import suppress
from pathlib import Path
from typing import Any, Final

# Bump when a derivation changes meaning; rows stamped with an older version
# are treated as index misses (stale) and re-derived from the analysis store.
INDEX_VERSION: Final[int] = 2

INDEX_FILENAME: Final[str] = "quick_index.sqlite3"

# Camelot key buckets: 0-23 = (camelot_number - 1) * 2 + (1 if letter == "B").
# This is the bucketed form of engine.camelot_affinity's tiers: two codes are
# compatible iff their buckets are equal or adjacent on the same ring, or share
# a number across rings (see compatible_buckets below).
MOOD_TIERS: Final[tuple[str, ...]] = ("euphoric", "peaceful", "tense", "melancholic")

ENERGY_BANDS: Final[tuple[str, ...]] = ("very_low", "low", "mid", "high", "very_high")

VOCAL_TIER_NAMES: Final[dict[int, str]] = {0: "instrumental", 1: "balanced", 2: "vocal"}

# vocal_activity is a per-bin 0-1 envelope (models/audio_analysis.py:82); its
# mean over the track discriminates instrumental (near-zero) from vocal-dense
# material. Thresholds are conservative: only a track with essentially no
# vocal energy anywhere is tier 0, because the engine's vocal gating (halved
# stretch limits, instrumental controls) errs toward treating a track as vocal.
VOCAL_DENSITY_INSTRUMENTAL: Final[float] = 0.02
VOCAL_DENSE_DENSITY: Final[float] = 0.25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_index (
    item_id TEXT NOT NULL,
    provider_instance TEXT NOT NULL,
    bpm REAL,
    bpm_norm REAL,
    bpm_half REAL,
    bpm_double REAL,
    key_bucket INTEGER,
    camelot TEXT,
    energy REAL,
    energy_tier INTEGER,
    energy_band TEXT,
    valence REAL,
    arousal REAL,
    mood_tier TEXT,
    mood TEXT,
    danceability REAL,
    loudness REAL,
    duration REAL,
    beats_per_bar INTEGER,
    vocal_density REAL,
    vocal_tier INTEGER,
    instrumental INTEGER,
    era TEXT,
    year INTEGER,
    analyzer_version INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (item_id, provider_instance)
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_qi_energy_tier ON track_index(energy_tier)",
    "CREATE INDEX IF NOT EXISTS ix_qi_key_bucket ON track_index(key_bucket)",
    "CREATE INDEX IF NOT EXISTS ix_qi_mood_tier ON track_index(mood_tier)",
    "CREATE INDEX IF NOT EXISTS ix_qi_era ON track_index(era)",
    "CREATE INDEX IF NOT EXISTS ix_qi_bpm_norm ON track_index(bpm_norm)",
)

# Column order matters: it defines the row shape returned by lookup/select.
_COLUMNS: Final[tuple[str, ...]] = (
    "item_id",
    "provider_instance",
    "bpm",
    "bpm_norm",
    "bpm_half",
    "bpm_double",
    "key_bucket",
    "camelot",
    "energy",
    "energy_tier",
    "energy_band",
    "valence",
    "arousal",
    "mood_tier",
    "mood",
    "danceability",
    "loudness",
    "duration",
    "beats_per_bar",
    "vocal_density",
    "vocal_tier",
    "instrumental",
    "era",
    "year",
    "analyzer_version",
    "updated_at",
)

_UPSERT_SQL = (
    f"INSERT INTO track_index ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _COLUMNS)}) "
    "ON CONFLICT(item_id, provider_instance) DO UPDATE SET "
    + ", ".join(f"{col} = excluded.{col}" for col in _COLUMNS[2:])
)


def _value(src: Any, name: str) -> Any:
    """Read ``name`` from either a mapping or an object (AudioAnalysisData)."""
    if isinstance(src, dict):
        return src.get(name)
    return getattr(src, name, None)


def _finite(value: Any) -> float | None:
    """Return value as a finite float, else None (vendor/model data is untrusted)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def _clamped(value: Any, lo: float, hi: float) -> float | None:
    """Return value as a finite float clamped into [lo, hi], else None."""
    as_float = _finite(value)
    if as_float is None:
        return None
    return min(hi, max(lo, as_float))


def bpm_window(bpm: Any) -> tuple[float | None, float | None, float | None]:
    """
    Fold a raw BPM into the 80-160 normalization window (spec 3.1).

    Returns (bpm_norm, bpm_half, bpm_double): bpm_norm is the raw bpm folded
    into [80, 160] by doubling/halving (87 <-> 174, 70 <-> 140), so 2:1
    aliasing across genres (hip-hop/trap half-time, DnB double-time) becomes
    an exact indexed match. Returns (None, None, None) for unusable bpm.
    """
    as_float = _finite(bpm)
    if as_float is None or as_float <= 0:
        return (None, None, None)
    norm = as_float
    while norm < 80.0:
        norm *= 2.0
    while norm > 160.0:
        norm /= 2.0
    return (norm, norm / 2.0, norm * 2.0)


def key_bucket(camelot: Any) -> int | None:
    """
    Map a Camelot code (e.g. "8A") to its 0-23 bucket: (num - 1) * 2 + is_b.

    Returns None for anything unparsable - never raise, the index must not
    reject a whole analysis because one key field is malformed.
    """
    if not isinstance(camelot, str) or len(camelot) < 2:
        return None
    digits, letter = camelot[:-1], camelot[-1].upper()
    try:
        number = int(digits)
    except ValueError:
        return None
    if not 1 <= number <= 12:
        return None
    if letter == "A":
        return (number - 1) * 2
    if letter == "B":
        return (number - 1) * 2 + 1
    return None


def compatible_buckets(camelot: Any) -> list[int]:
    """
    Return every key bucket compatible with the given Camelot code.

    "Compatible" = engine.camelot_affinity > 0.0 (engine.py tiers 1.0 / 0.85 /
    0.72 / 0.55 / 0.0): the code itself, +/-1 on the same ring, the same
    number on the opposite ring, and +/-1 on the opposite ring - at most 6
    buckets (spec 3.3). Unparsable input yields an empty list (no filter).
    """
    bucket = key_bucket(camelot)
    if bucket is None:
        return []
    number = bucket // 2 + 1
    letter = "B" if bucket % 2 else "A"
    other = "B" if letter == "A" else "A"
    out = {bucket}
    for delta in (1, 11):  # ring-neighbours modulo 12
        neighbor = (number + delta - 1) % 12 + 1
        out.add((neighbor - 1) * 2 + (0 if letter == "A" else 1))
    out.add((number - 1) * 2 + (1 if other == "B" else 0))
    for delta in (1, 11):
        neighbor = (number + delta - 1) % 12 + 1
        out.add((neighbor - 1) * 2 + (1 if other == "B" else 0))
    return sorted(out)


def energy_tier(energy: Any) -> int | None:
    """Bucket 0-1 energy into 5 tiers: min(4, floor(energy * 5)) (spec 3.1)."""
    as_float = _clamped(energy, 0.0, 1.0)
    if as_float is None:
        return None
    return min(4, int(as_float * 5))


def energy_band(tier: Any) -> str | None:
    """Name an energy tier; None passes through (unknown tier, no band)."""
    as_int = tier if isinstance(tier, int) and not isinstance(tier, bool) else None
    if as_int is None or not 0 <= as_int <= 4:
        return None
    return ENERGY_BANDS[as_int]


def mood_tier(valence: Any, arousal: Any) -> str | None:
    """
    Map the valence/arousal plane to a 4-quadrant mood tier (playbook Part III).

    valence is the musical mood 0.0 (dark/sad) to 1.0 (bright/happy) and
    arousal the intensity 0.0 (calm) to 1.0 (energetic) - both already stored
    by the local analyzer (models/audio_analysis.py:119-123). Missing either
    means "unknown", not "the middle".
    """
    val = _clamped(valence, 0.0, 1.0)
    aro = _clamped(arousal, 0.0, 1.0)
    if val is None or aro is None:
        return None
    if val >= 0.5:
        return "euphoric" if aro >= 0.5 else "peaceful"
    return "tense" if aro >= 0.5 else "melancholic"


def era_bucket(year: Any) -> str | None:
    """Bucket a release year into a decade label ("1987" -> "1980s")."""
    as_int = year if isinstance(year, int) and not isinstance(year, bool) else None
    if as_int is None or not 1000 <= as_int <= 3000:
        return None
    return f"{(as_int // 10) * 10}s"


def vocal_tier(density: Any) -> int | None:
    """
    Bucket mean vocal_activity into 0=instrumental / 1=balanced / 2=vocal.

    None only when the density itself is unknown.
    """
    as_float = _clamped(density, 0.0, 1.0)
    if as_float is None:
        return None
    if as_float < VOCAL_DENSITY_INSTRUMENTAL:
        return 0
    if as_float >= VOCAL_DENSE_DENSITY:
        return 2
    return 1


def _mean_envelope(values: Any) -> float | None:
    """Mean of the finite numbers in an envelope list, else None."""
    if not isinstance(values, (list, tuple)):
        return None
    total = 0.0
    count = 0
    for item in values:
        as_float = _finite(item)
        if as_float is not None:
            total += as_float
            count += 1
    if not count:
        return None
    return total / count


def camelot_from_key(key: str | None, mode: str | None) -> str | None:
    """Convert Music Assistant key/mode to Camelot notation."""
    if not key or not mode:
        return None
    minor_keys = {
        "Ab": "1A",
        "Eb": "2A",
        "Bb": "3A",
        "F": "4A",
        "C": "5A",
        "G": "6A",
        "D": "7A",
        "A": "8A",
        "E": "9A",
        "B": "10A",
        "F#": "11A",
        "C#": "12A",
    }
    major_keys = {
        "B": "1B",
        "F#": "2B",
        "C#": "3B",
        "Ab": "4B",
        "Eb": "5B",
        "Bb": "6B",
        "F": "7B",
        "C": "8B",
        "G": "9B",
        "D": "10B",
        "A": "11B",
        "E": "12B",
    }
    minor = mode.lower() in {"minor", "min", "m"}
    normalized = key.replace("♭", "b").replace("♯", "#")
    return (minor_keys if minor else major_keys).get(normalized)


def metadata_facts(metadata: Any) -> dict[str, Any]:
    """
    Extract the MA metadata facts the index derives from: mood and year.

    ``metadata`` is a music_assistant_models MediaItemMetadata (mood at
    metadata.py:191, release_date at :207 - frequently None, provider
    populated). Returns {} when nothing is usable; never raises.
    """
    facts: dict[str, Any] = {}
    mood = _value(metadata, "mood")
    if isinstance(mood, str) and mood.strip():
        facts["mood"] = mood.strip().lower()
    release = _value(metadata, "release_date")
    if release is not None:
        with suppress(AttributeError, TypeError, ValueError):
            facts["year"] = int(release.year)
    return facts


def derive_row(
    item_id: str,
    provider_instance: str,
    values: Any,
    metadata: Any = None,
) -> dict[str, Any]:
    """
    Derive one quick-pull index row from stored analysis (+ optional metadata).

    ``values`` is an AudioAnalysisData (models/audio_analysis.py:55+) or a
    dict of the same field names; ``metadata`` is the track's
    MediaItemMetadata when the caller has one. Only DERIVED scalars land in
    the row - no beats, no envelopes, no embeddings.
    """
    bpm_raw = _finite(_value(values, "bpm"))
    if bpm_raw is not None and bpm_raw <= 0:
        bpm_raw = None
    bpm_norm, bpm_half, bpm_double = bpm_window(bpm_raw)
    camelot = _value(values, "camelot") or camelot_from_key(
        _value(values, "key"), _value(values, "mode")
    )
    energy = _clamped(_value(values, "energy"), 0.0, 1.0)
    tier = energy_tier(energy)
    valence = _clamped(_value(values, "valence"), 0.0, 1.0)
    arousal = _clamped(_value(values, "arousal"), 0.0, 1.0)
    vocal_density = _mean_envelope(_value(values, "vocal_activity"))
    instrumentalness = _clamped(_value(values, "instrumentalness"), 0.0, 1.0)
    if instrumentalness is not None:
        # same derivation as the provider's MA analysis branch (>= 0.5)
        instrumental: int | None = int(instrumentalness >= 0.5)
    else:
        instrumental = None
    facts = metadata_facts(metadata)
    year = facts.get("year")
    beats_per_bar = _value(values, "beats_per_bar")
    bpb = int(beats_per_bar) if isinstance(beats_per_bar, int) and beats_per_bar > 0 else None
    return {
        "item_id": str(item_id),
        "provider_instance": str(provider_instance),
        "bpm": bpm_raw,
        "bpm_norm": bpm_norm,
        "bpm_half": bpm_half,
        "bpm_double": bpm_double,
        "key_bucket": key_bucket(camelot),
        "camelot": camelot,
        "energy": energy,
        "energy_tier": tier,
        "energy_band": energy_band(tier),
        "valence": valence,
        "arousal": arousal,
        "mood_tier": mood_tier(valence, arousal),
        "mood": facts.get("mood"),
        "danceability": _clamped(_value(values, "danceability"), 0.0, 1.0),
        "loudness": _finite(_value(values, "loudness_integrated")),
        "duration": _finite(_value(values, "duration")),
        "beats_per_bar": bpb,
        "vocal_density": vocal_density,
        "vocal_tier": vocal_tier(vocal_density),
        "instrumental": instrumental,
        "era": era_bucket(year),
        "year": year,
        "analyzer_version": INDEX_VERSION,
        "updated_at": time_module.time(),
    }


def analysis_from_index_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Rebuild a planner-facing analysis dict from one derived index row.

    This is the fast path's stand-in for a full ``_analysis`` result: every
    SCALAR the beam optimizer scores on (bpm, camelot, energy, danceability,
    loudness, instrumental) is present, but the raw arrays (beats, rms_energy,
    clap_embedding) are not - the index is derived-only by design, so the
    array-backed scoring signals (quiet-edge segue overlap, sonic similarity)
    are simply absent for index-sourced rows instead of being wrong.
    """
    instrumental = row.get("instrumental")
    return {
        "bpm": row.get("bpm"),
        "camelot": row.get("camelot"),
        "energy": row.get("energy"),
        "danceability": row.get("danceability"),
        "valence": row.get("valence"),
        "arousal": row.get("arousal"),
        "loudness": row.get("loudness"),
        "beats_per_bar": row.get("beats_per_bar") or 4,
        "instrumental": None if instrumental is None else bool(instrumental),
        "vocal_density": row.get("vocal_density"),
        "mood_tier": row.get("mood_tier"),
        "mood": row.get("mood"),
        "era": row.get("era"),
        "year": row.get("year"),
        "source": "quick_index",
        "index_derived": True,
    }


def _tempo_key_conditions(
    bpm: float | None,
    bpm_tolerance: float,
    key: str | None,
) -> tuple[list[str], list[Any]]:
    """
    Build the bpm and key WHERE fragments for a quick-pull query.

    bpm matches the normalized 80-160 window OR the 2:1 aliases
    (bpm_half/bpm_double) - the half-time/double-time aliasing the genre
    briefs demand (87 <-> 174, 70 <-> 140). key expands to the compatible
    Camelot buckets (at most 6, per camelot_affinity). Raises ValueError
    naming the offending filter.
    """
    conditions: list[str] = []
    params: list[Any] = []
    if bpm is not None:
        as_float = _finite(bpm)
        if as_float is None or as_float <= 0:
            raise ValueError(f"bpm={bpm!r}")
        tol = _finite(bpm_tolerance)
        tol = 0.06 if tol is None else min(0.5, max(0.01, tol))
        lo, hi = as_float * (1 - tol), as_float * (1 + tol)
        branches = []
        for column in ("bpm_norm", "bpm_half", "bpm_double"):
            branches.append(f"({column} BETWEEN ? AND ?)")
            params.extend((lo, hi))
        conditions.append("(" + " OR ".join(branches) + ")")
    if key is not None:
        buckets = compatible_buckets(key)
        if not buckets:
            raise ValueError(f"key={key!r}")
        conditions.append(f"key_bucket IN ({', '.join('?' for _ in buckets)})")
        params.extend(buckets)
    return conditions, params


def _tier_conditions(
    energy: float | None,
    energy_band: str | None,
    vocal: int | str | None,
) -> tuple[list[str], list[Any]]:
    """Build the energy/vocal tier WHERE fragments for a quick-pull query."""
    conditions: list[str] = []
    params: list[Any] = []
    if energy is not None:
        as_float = _finite(energy)
        if as_float is None or not 0.0 <= as_float <= 1.0:
            raise ValueError(f"energy={energy!r}")
        tier = energy_tier(energy)
        if tier is None:  # unreachable: energy is validated in [0, 1] above
            raise ValueError(f"energy={energy!r}")
        conditions.append("energy_tier BETWEEN ? AND ?")
        params.extend((max(0, tier - 1), min(4, tier + 1)))
    if energy_band is not None:
        if energy_band not in ENERGY_BANDS:
            raise ValueError(f"energy_band={energy_band!r}")
        conditions.append("energy_tier = ?")
        params.append(ENERGY_BANDS.index(energy_band))
    if vocal is not None:
        tier_value: int | None = None
        if isinstance(vocal, bool):
            tier_value = None
        elif isinstance(vocal, int) and vocal in VOCAL_TIER_NAMES:
            tier_value = vocal
        elif isinstance(vocal, str):
            lowered = vocal.strip().lower()
            for tier_num, name in VOCAL_TIER_NAMES.items():
                if name == lowered:
                    tier_value = tier_num
                    break
        if tier_value is None:
            raise ValueError(f"vocal={vocal!r}")
        conditions.append("vocal_tier = ?")
        params.append(tier_value)
    return conditions, params


def _metadata_conditions(
    mood: str | None,
    era: str | None,
    year: int | None,
    exclude_ids: list[str] | None,
) -> tuple[list[str], list[Any]]:
    """Build the mood/era/year/exclusion WHERE fragments for a quick-pull query."""
    conditions: list[str] = []
    params: list[Any] = []
    if mood is not None:
        if not isinstance(mood, str) or not mood.strip():
            raise ValueError(f"mood={mood!r}")
        lowered = mood.strip().lower()
        if lowered in MOOD_TIERS:
            conditions.append("mood_tier = ?")
            params.append(lowered)
        else:
            # fall through to the MA metadata mood string (metadata.py:191)
            conditions.append("mood = ?")
            params.append(lowered)
    if era is not None:
        if not isinstance(era, str) or not era.strip().endswith("s"):
            raise ValueError(f"era={era!r}")
        conditions.append("era = ?")
        params.append(era.strip().lower())
    if year is not None:
        if not isinstance(year, int) or isinstance(year, bool) or not 1000 <= year <= 3000:
            raise ValueError(f"year={year!r}")
        conditions.append("year = ?")
        params.append(year)
    if exclude_ids:
        excluded = [str(item) for item in exclude_ids][:500]
        if excluded:
            conditions.append(f"item_id NOT IN ({', '.join('?' for _ in excluded)})")
            params.extend(excluded)
    return conditions, params


def _select_conditions(
    *,
    bpm: float | None,
    bpm_tolerance: float,
    key: str | None,
    energy: float | None,
    energy_band: str | None,
    mood: str | None,
    era: str | None,
    year: int | None,
    vocal: int | str | None,
    exclude_ids: list[str] | None,
) -> tuple[list[str], list[Any]]:
    """
    Build the WHERE clause for a quick-pull query from validated filters.

    Every condition is parameterized; no filter value is ever interpolated.
    Raises ValueError naming the offending filter.
    """
    conditions, params = _tempo_key_conditions(bpm, bpm_tolerance, key)
    tier_conds, tier_params = _tier_conditions(energy, energy_band, vocal)
    meta_conds, meta_params = _metadata_conditions(mood, era, year, exclude_ids)
    conditions.extend(tier_conds)
    params.extend(tier_params)
    conditions.extend(meta_conds)
    params.extend(meta_params)
    return conditions, params


class QuickIndex:
    """
    Thin synchronous SQLite wrapper for the derived quick-pull index.

    Every operation opens its own short-lived connection (the index is tiny
    and write rates are one row per stored analysis), which sidesteps
    cross-thread connection issues when calls are bridged with
    ``asyncio.to_thread``. A class-level lock serializes writers per index
    file. All methods are synchronous; the provider wraps them in threads.
    """

    _write_locks: Final[dict[str, threading.Lock]] = {}

    def __init__(self, db_path: Path | str) -> None:
        """Point the index at a file path; the schema opens lazily."""
        self.db_path = Path(db_path)
        self._opened = False

    def all_keys(self, version: int | None = None) -> set[tuple[str, str]]:
        """Every (item_id, provider_instance) key in the index (backfill skip-set)."""
        where, params = "", []
        if version is not None:
            where = " WHERE analyzer_version = ?"
            params.append(version)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT item_id, provider_instance FROM track_index{where}", params
            ).fetchall()
        return {(str(row[0]), str(row[1])) for row in rows}

    def count(self) -> int:
        """Return the number of rows currently indexed (any version)."""
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM track_index").fetchone()[0])

    @property
    def is_open(self) -> bool:
        """True once the schema has been ensured."""
        return self._opened

    def lookup(
        self,
        keys: list[tuple[str, str]],
        version: int | None = INDEX_VERSION,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Batch-fetch index rows by (item_id, provider_instance) key.

        Rows stamped with a different analyzer_version are treated as misses
        (stale) when ``version`` is given, so callers fall back to the full
        analysis path instead of mixing derivation semantics.
        """
        if not keys:
            return {}
        params: list[Any] = []
        clauses = []
        for chunk_start in range(0, len(keys), 500):
            chunk = keys[chunk_start : chunk_start + 500]
            placeholders = []
            for item_id, provider_instance in chunk:
                placeholders.append("(item_id = ? AND provider_instance = ?)")
                params.extend((str(item_id), str(provider_instance)))
            clauses.append("(" + " OR ".join(placeholders) + ")")
        where = " OR ".join(clauses)
        if version is not None:
            where = f"({where}) AND analyzer_version = ?"
            params.append(version)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM track_index WHERE {where}", params
            ).fetchall()
        return {
            (row["item_id"], row["provider_instance"]): row for row in map(self._row_to_dict, rows)
        }

    def open(self) -> None:
        """Create the schema if missing. Idempotent and cheap enough to re-call."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(), self._connect() as conn:
            conn.execute(_SCHEMA)
            for statement in _INDEXES:
                conn.execute(statement)
            # one-time purge of rows stamped by an older INDEX_VERSION: v1
            # shipped before the tombstone gate, so pre-bump rows can carry
            # all-null shells that count() and select() would keep serving
            conn.execute(
                "DELETE FROM track_index WHERE analyzer_version < ?",
                (INDEX_VERSION,),
            )
        self._opened = True

    def select(  # noqa: PLR0913 - the filter surface mirrors the command's
        self,
        *,
        bpm: float | None = None,
        bpm_tolerance: float = 0.06,
        key: str | None = None,
        energy: float | None = None,
        energy_band: str | None = None,
        mood: str | None = None,
        era: str | None = None,
        year: int | None = None,
        vocal: int | str | None = None,
        exclude_ids: list[str] | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Query the index with the pull_candidates filter surface.

        Every parameter is optional; conditions AND together. Raises
        ValueError on filter values that cannot be honored; callers translate
        that into their own error surface. See ``_select_conditions`` for the
        per-filter semantics.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit={limit!r}")
        conditions, params = _select_conditions(
            bpm=bpm,
            bpm_tolerance=bpm_tolerance,
            key=key,
            energy=energy,
            energy_band=energy_band,
            mood=mood,
            era=era,
            year=year,
            vocal=vocal,
            exclude_ids=exclude_ids,
        )
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM track_index{where} "
                f"ORDER BY updated_at DESC, item_id LIMIT ?",
                [*params, limit],
            ).fetchall()
        return list(map(self._row_to_dict, rows))

    def upsert(self, row: dict[str, Any]) -> None:
        """Upsert one derived row."""
        self.upsert_many([row])

    def upsert_many(self, rows: list[dict[str, Any]]) -> int:
        """Upsert derived rows; returns the number of rows written."""
        if not rows:
            return 0
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            tuple(row.get(col) for col in _COLUMNS)
            for row in rows
            if row.get("item_id") and row.get("provider_instance")
        ]
        with self._lock(), self._connect() as conn:
            conn.execute(_SCHEMA)
            for statement in _INDEXES:
                conn.execute(statement)
            conn.executemany(_UPSERT_SQL, payload)
        self._opened = True
        return len(payload)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _lock(self) -> threading.Lock:
        return QuickIndex._write_locks.setdefault(str(self.db_path), threading.Lock())

    def _row_to_dict(self, db_row: tuple[Any, ...]) -> dict[str, Any]:
        return dict(zip(_COLUMNS, db_row, strict=True))
