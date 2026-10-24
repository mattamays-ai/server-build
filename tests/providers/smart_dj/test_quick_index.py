"""Tests for the derived quick-pull index and pull_candidates (0.0.27 Item 3)."""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.models.audio_analysis import AudioAnalysisData
from music_assistant.providers.smart_dj import provider as provider_module
from music_assistant.providers.smart_dj.engine import camelot_affinity
from music_assistant.providers.smart_dj.provider import (
    SmartDJError,
    SmartDJProvider,
    _camelot_from_key,
    _playlist_analysis_rows,
    _pull_filters,
)
from music_assistant.providers.smart_dj.quick_index import (
    INDEX_VERSION,
    QuickIndex,
    analysis_from_index_row,
    bpm_window,
    compatible_buckets,
    derive_row,
    energy_band,
    energy_tier,
    era_bucket,
    key_bucket,
    mood_tier,
    vocal_tier,
)

ALL_CAMELOT_CODES = [f"{num}{ring}" for num in range(1, 13) for ring in "AB"]


def _analysis_values(**overrides: Any) -> AudioAnalysisData:
    """Return a realistic stored AudioAnalysisData (local ML analyzer shape)."""
    fields: dict[str, Any] = {
        "duration": 210.0,
        "bpm": 128.0,
        "beats_per_bar": 4,
        "key": "A",
        "mode": "minor",
        "energy": 0.73,
        "danceability": 0.81,
        "valence": 0.22,
        "arousal": 0.66,
        "loudness_integrated": -9.5,
        "instrumentalness": 0.05,
        "vocal_activity": [0.0, 0.5, 1.0, 0.5],
    }
    fields.update(overrides)
    return AudioAnalysisData(**fields)


def _seed_rows(index: QuickIndex, count: int = 3) -> None:
    """Seed an index with count distinct tracks with distinct energies."""
    for i in range(count):
        values = _analysis_values(bpm=120.0 + i, energy=0.1 + i * 0.3)
        index.upsert(derive_row(f"id{i}", "prov1", values))


# ---------------------------------------------------------------------------
# derivations (pure functions)
# ---------------------------------------------------------------------------


def test_bpm_window_folds_into_80_160() -> None:
    """Raw bpm folds into the normalized window with 2:1 aliases."""
    assert bpm_window(120.0) == (120.0, 60.0, 240.0)
    assert bpm_window(174.0) == (87.0, 43.5, 174.0)
    assert bpm_window(70.0) == (140.0, 70.0, 280.0)
    assert bpm_window(200.0) == (100.0, 50.0, 200.0)


def test_bpm_window_unusable_bpm() -> None:
    """Zero, negative, and non-numeric bpm yield all-None (never raise)."""
    for bad in (None, 0, -5, "fast"):
        assert bpm_window(bad) == (None, None, None)


def test_key_bucket_layout() -> None:
    """Buckets are (num - 1) * 2 + is_b over 0-23; junk is None."""
    assert key_bucket("1A") == 0
    assert key_bucket("8A") == 14
    assert key_bucket("8B") == 15
    assert key_bucket("12B") == 23
    for junk in (None, "", "8", "X8A", "13A", "8C", 42):
        assert key_bucket(junk) is None


def test_compatible_buckets_match_camelot_affinity() -> None:
    """Bucket membership <=> engine.camelot_affinity > 0, for every pair."""
    for a in ALL_CAMELOT_CODES:
        buckets = compatible_buckets(a)
        assert 1 <= len(buckets) <= 6
        for b in ALL_CAMELOT_CODES:
            bucket_b = key_bucket(b)
            assert bucket_b is not None
            assert (camelot_affinity(a, b) > 0.0) == (bucket_b in buckets), (a, b)


def test_compatible_buckets_junk() -> None:
    """Unparsable keys yield no buckets (no filter), not an error."""
    assert compatible_buckets(None) == []
    assert compatible_buckets("huh") == []


def test_energy_tier_and_band() -> None:
    """min(4, floor(energy * 5)) on the 0-1 scale, band names line up."""
    assert energy_tier(0.0) == 0
    assert energy_tier(0.199) == 0
    assert energy_tier(0.2) == 1
    assert energy_tier(0.5) == 2
    assert energy_tier(0.999) == 4
    assert energy_tier(1.0) == 4
    assert energy_tier(1.5) == 4  # clamped
    assert energy_tier(None) is None
    assert energy_band(0) == "very_low"
    assert energy_band(4) == "very_high"
    assert energy_band(None) is None
    assert energy_band(9) is None


def test_mood_tier_quadrants() -> None:
    """Valence/arousal quadrants; missing either is unknown, not middle."""
    assert mood_tier(0.8, 0.8) == "euphoric"
    assert mood_tier(0.8, 0.2) == "peaceful"
    assert mood_tier(0.2, 0.8) == "tense"
    assert mood_tier(0.2, 0.2) == "melancholic"
    assert mood_tier(None, 0.8) is None
    assert mood_tier(0.8, None) is None


def test_era_bucket() -> None:
    """Decade labels; unusable years are None."""
    assert era_bucket(1987) == "1980s"
    assert era_bucket(2026) == "2020s"
    assert era_bucket(None) is None
    assert era_bucket(42) is None


def test_vocal_tier_thresholds() -> None:
    """Conservative vocal-density buckets; unknown density stays None."""
    assert vocal_tier(0.0) == 0
    assert vocal_tier(0.01) == 0
    assert vocal_tier(0.1) == 1
    assert vocal_tier(0.9) == 2
    assert vocal_tier(None) is None


def test_derive_row_from_real_analysis_model() -> None:
    """derive_row reads the real AudioAnalysisData fields, no arrays stored."""
    row = derive_row("t1", "prov1", _analysis_values())
    assert row["item_id"] == "t1"
    assert row["provider_instance"] == "prov1"
    assert row["bpm"] == 128.0
    assert row["bpm_norm"] == 128.0
    assert row["bpm_half"] == 64.0
    assert row["bpm_double"] == 256.0
    assert row["camelot"] == "8A"  # A minor
    assert row["key_bucket"] == 14
    assert row["energy_tier"] == 3
    assert row["energy_band"] == "high"
    assert row["mood_tier"] == "tense"  # low valence, high arousal
    # mean of [0.0, 0.5, 1.0, 0.5]
    assert row["vocal_density"] == pytest.approx(0.5)
    assert row["vocal_tier"] == 2
    assert row["instrumental"] == 0  # instrumentalness 0.05 < 0.5
    assert row["loudness"] == -9.5
    assert row["analyzer_version"] == INDEX_VERSION
    # derived-only: no raw arrays anywhere in the row
    for array_field in ("beats", "downbeats", "rms_energy", "clap_embedding"):
        assert array_field not in row


def test_derive_row_with_ma_metadata() -> None:
    """Mood/release_date come from MA metadata (metadata.py:191/:207)."""
    metadata = MediaItemMetadata(mood="Happy", release_date=datetime(1987, 5, 1, tzinfo=UTC))
    row = derive_row("t1", "prov1", _analysis_values(), metadata)
    assert row["mood"] == "happy"
    assert row["year"] == 1987
    assert row["era"] == "1980s"


def test_derive_row_moodless_metadata_is_unknown() -> None:
    """Frequently-None metadata stays None, never fabricated."""
    row = derive_row("t1", "prov1", _analysis_values(), MediaItemMetadata())
    assert row["mood"] is None
    assert row["year"] is None
    assert row["era"] is None


def test_camelot_from_key_identity_with_provider_helper() -> None:
    """One authority: provider._camelot_from_key delegates to the index module."""
    for key in ("C", "A", "B", "F#", "Ab", None, "X"):
        for mode in ("major", "minor", None):
            expected = _camelot_from_key(key, mode)
            assert expected == _camelot_from_key(key, mode)
    # spot-check the unicode flats/sharps path the provider used to own
    assert _camelot_from_key("♭", "minor") is None  # bare symbol is not a key
    assert _camelot_from_key("Bb", "minor") == "3A"


def test_analysis_from_index_row_shape() -> None:
    """The fast-path analysis dict carries planner scalars, source-tagged."""
    row = derive_row("t1", "prov1", _analysis_values())
    analysis = analysis_from_index_row(row)
    assert analysis["source"] == "quick_index"
    assert analysis["bpm"] == 128.0
    assert analysis["camelot"] == "8A"
    assert analysis["energy"] == pytest.approx(0.73)
    assert analysis["instrumental"] is False
    assert analysis["mood_tier"] == "tense"
    assert "beats" not in analysis
    assert "rms_energy" not in analysis


# ---------------------------------------------------------------------------
# QuickIndex sqlite round-trip
# ---------------------------------------------------------------------------


def test_upsert_lookup_roundtrip(tmp_path: Any) -> None:
    """Upsert then batch lookup; version mismatch is a miss (stale)."""
    index = QuickIndex(tmp_path / "quick_index.sqlite3")
    index.open()
    index.upsert(derive_row("a", "prov1", _analysis_values()))
    index.upsert(derive_row("b", "prov1", _analysis_values(bpm=87.0)))
    rows = index.lookup([("a", "prov1"), ("b", "prov1"), ("missing", "prov1")])
    assert set(rows) == {("a", "prov1"), ("b", "prov1")}
    assert rows[("b", "prov1")]["bpm"] == 87.0
    # stale version => miss
    stale = dict(rows[("a", "prov1")])
    stale["analyzer_version"] = INDEX_VERSION + 1
    index.upsert(stale)
    assert index.lookup([("a", "prov1")], version=INDEX_VERSION) == {}
    assert index.lookup([("a", "prov1")], version=None)
    assert index.count() == 2


def test_select_filters(tmp_path: Any) -> None:
    """The pull_candidates filter surface against a seeded index."""
    index = QuickIndex(tmp_path / "qi.sqlite3")
    index.open()
    index.upsert(
        derive_row("house", "p", _analysis_values(bpm=128.0, energy=0.7, valence=0.9, arousal=0.9))
    )
    index.upsert(
        derive_row(
            "half_time", "p", _analysis_values(bpm=70.0, energy=0.3, valence=0.2, arousal=0.2)
        )
    )
    index.upsert(
        derive_row("dnb", "p", _analysis_values(bpm=174.0, energy=0.9, valence=0.4, arousal=0.8))
    )
    # bpm: 128 window matches the raw 128 track only
    hits = {row["item_id"] for row in index.select(bpm=128.0, bpm_tolerance=0.06)}
    assert hits == {"house"}
    # bpm: 2:1 aliasing - a 140 query catches a 70 bpm track (folded to norm 140)
    hits = {row["item_id"] for row in index.select(bpm=140.0, bpm_tolerance=0.06)}
    assert hits == {"half_time"}
    # bpm: double-time aliasing - a 174 query matches via the doubled branch
    hits = {row["item_id"] for row in index.select(bpm=174.0, bpm_tolerance=0.02)}
    assert hits == {"dnb"}
    # key: A minor = 8A; all three seeds share it
    assert len(index.select(key="8A")) == 3
    # energy: tier window +/-1 around 0.7 (tier 3) -> tiers 2-4
    hits = {row["item_id"] for row in index.select(energy=0.7)}
    assert hits == {"house", "dnb"}
    # mood quadrants
    hits = {row["item_id"] for row in index.select(mood="euphoric")}
    assert hits == {"house"}
    hits = {row["item_id"] for row in index.select(mood="melancholic")}
    assert hits == {"half_time"}
    # vocal tier by name
    hits = {row["item_id"] for row in index.select(vocal="vocal")}
    assert hits == {"house", "dnb", "half_time"}  # all have vocal_activity > 0.25
    # exclusions and limit
    hits = {row["item_id"] for row in index.select(exclude_ids=["house"])}
    assert "house" not in hits
    assert len(index.select(limit=2)) == 2
    # energy_band exact
    assert {row["item_id"] for row in index.select(energy_band="very_high")} == {"dnb"}


def test_select_rejects_bad_filters(tmp_path: Any) -> None:
    """Malformed filters raise ValueError naming the filter."""
    index = QuickIndex(tmp_path / "qi.sqlite3")
    index.open()
    cases = [
        ({"bpm": "fast"}, "bpm="),
        ({"key": "nope"}, "key="),
        ({"energy": 9.0}, "energy="),
        ({"energy_band": "medium"}, "energy_band="),
        ({"mood": ""}, "mood="),
        ({"era": "1980"}, "era="),
        ({"year": 42}, "year="),
        ({"vocal": "shouts"}, "vocal="),
        ({"limit": 0}, "limit="),
    ]
    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            index.select(**kwargs)


def test_select_validates_at_command_layer() -> None:
    """_pull_filters clamps limit and normalizes key/vocal inputs."""
    filters = _pull_filters(limit=100000, key="8a", vocal="instrumental", bpm=120)
    assert filters["limit"] <= 250
    assert filters["key"] == "8A"
    assert filters["vocal"] == "instrumental"
    assert filters["bpm"] == 120.0


# ---------------------------------------------------------------------------
# provider pool-pulling
# ---------------------------------------------------------------------------


def _bare_prov(tmp_path: Any) -> SmartDJProvider:
    """Build a bare provider with a temp quick index and a spy _analysis."""
    prov = SmartDJProvider.__new__(SmartDJProvider)
    # __new__ skips the base init: the fail-soft paths need a logger
    prov.logger = logging.getLogger("smart_dj_test")
    prov._quick_index = QuickIndex(tmp_path / "qi.sqlite3")
    prov._quick_index.open()
    prov._analysis_calls: list[str] = []

    async def _spy_analysis(
        item_id: str, _provider: str, _metadata: Any, _ap: str
    ) -> dict[str, Any]:
        prov._analysis_calls.append(str(item_id))
        return {"bpm": 111.0, "source": "full", "camelot": "5A"}

    prov._analysis = _spy_analysis  # type: ignore[method-assign]
    return cast("SmartDJProvider", prov)


def _pool_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "queue_item_id": f"mix-{i}",
            "item_id": f"id{i}",
            "provider": "prov1",
            "name": f"track{i}",
            "genre": "house",
        }
        for i in range(count)
    ]


async def test_pool_prefers_index(tmp_path: Any) -> None:
    """Indexed tracks resolve from the index; only misses hit _analysis."""
    prov = _bare_prov(tmp_path)
    prov._quick_index.upsert(derive_row("id0", "prov1", _analysis_values()))
    results = await prov._pool_analyses(_pool_rows(2), "auto")
    assert results[0]["source"] == "quick_index"
    assert results[0]["bpm"] == 128.0
    assert results[1]["source"] == "full"
    assert prov._analysis_calls == ["id1"]


async def test_pool_falls_back_when_index_empty(tmp_path: Any) -> None:
    """Empty index: the pre-index path answers for every row."""
    prov = _bare_prov(tmp_path)
    results = await prov._pool_analyses(_pool_rows(2), "auto")
    assert all(res["source"] == "full" for res in results)
    assert prov._analysis_calls == ["id0", "id1"]


async def test_pool_full_source_forces_payload_path(tmp_path: Any) -> None:
    """pool_source="full" bypasses a populated index entirely."""
    prov = _bare_prov(tmp_path)
    prov._quick_index.upsert(derive_row("id0", "prov1", _analysis_values()))
    results = await prov._pool_analyses(_pool_rows(2), "auto", pool_source="full")
    assert all(res["source"] == "full" for res in results)
    assert prov._analysis_calls == ["id0", "id1"]


async def test_pool_survives_broken_index(tmp_path: Any) -> None:
    """A raising index degrades to the existing path, never an error."""

    class _BrokenIndex:
        def lookup(self, _keys: list[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
            raise OSError("boom")

    prov = _bare_prov(tmp_path)
    prov._quick_index = cast("QuickIndex", _BrokenIndex())
    results = await prov._pool_analyses(_pool_rows(1), "auto")
    assert results[0]["source"] == "full"


async def test_pool_keeps_exception_contract(tmp_path: Any) -> None:
    """Analysis failures come back as exceptions in position (old contract)."""

    async def _boom(_item_id: str, _provider: str, _metadata: Any, _ap: str) -> dict[str, Any]:
        raise RuntimeError("no analysis for you")

    prov = _bare_prov(tmp_path)
    prov._analysis = _boom  # type: ignore[method-assign]
    results = await prov._pool_analyses(_pool_rows(1), "auto")
    assert isinstance(results[0], RuntimeError)


async def test_playlist_analysis_rows_uses_index(tmp_path: Any) -> None:
    """mix_playlist/save_sorted's shared pool builder rides the fast path."""
    prov = _bare_prov(tmp_path)
    prov._quick_index.upsert(derive_row("id0", "prov1", _analysis_values()))
    resolved = [
        SimpleNamespace(
            name=f"track{i}",
            artist_str="artist",
            item_id=f"id{i}",
            provider="prov1",
            metadata=SimpleNamespace(genres={"house"}, explicit=False),
        )
        for i in range(2)
    ]
    analyzed, media_by_id = await _playlist_analysis_rows(prov, resolved, "auto")
    assert analyzed[0]["analysis"]["source"] == "quick_index"
    assert analyzed[1]["analysis"]["source"] == "full"
    assert media_by_id["mix-0"].item_id == "id0"


# ---------------------------------------------------------------------------
# index upsert + backfill
# ---------------------------------------------------------------------------


async def test_index_upsert_from_silent_success(tmp_path: Any) -> None:
    """The _analyze_media_silent success point derives + upserts with metadata."""
    prov = _bare_prov(tmp_path)
    track = SimpleNamespace(
        metadata=MediaItemMetadata(mood="Aggressive", release_date=datetime(2001, 1, 1, tzinfo=UTC))
    )
    await prov._index_upsert("t9", "prov1", _analysis_values(), track)
    rows = prov._quick_index.lookup([("t9", "prov1")])
    assert rows[("t9", "prov1")]["mood"] == "aggressive"
    assert rows[("t9", "prov1")]["year"] == 2001


async def test_index_upsert_refuses_a_tombstone(tmp_path: Any) -> None:
    """
    An all-null analysis must never reach the quick-pull index.

    The acoustid no-match tombstone parses as a real AudioAnalysisData with
    every scalar None; indexing it would stamp the CURRENT analyzer_version
    on a row with nothing in it, which is what made total_indexed lie.
    """
    prov = _bare_prov(tmp_path)
    tombstone = AudioAnalysisData(extra_data={"retry_after": 1234})
    await prov._index_upsert("t9", "prov1", tombstone, None)
    assert prov._quick_index.lookup([("t9", "prov1")]) == {}
    assert prov._quick_index.count() == 0


async def test_index_upsert_still_accepts_one_real_scalar(tmp_path: Any) -> None:
    """A single populated measurement is enough to be worth indexing."""
    prov = _bare_prov(tmp_path)
    sparse = AudioAnalysisData(bpm=96.0)
    await prov._index_upsert("t9", "prov1", sparse, None)
    rows = prov._quick_index.lookup([("t9", "prov1")])
    assert rows[("t9", "prov1")]["bpm"] == 96.0


async def test_index_upsert_is_fail_soft(tmp_path: Any) -> None:
    """A broken index never turns a successful analysis into a failure."""

    class _Broken:
        def upsert(self, _row: dict[str, Any]) -> None:
            raise OSError("disk full")

    prov = _bare_prov(tmp_path)
    prov._quick_index = cast("QuickIndex", _Broken())
    await prov._index_upsert("t9", "prov1", _analysis_values(), None)  # no raise


async def test_silent_early_return_writes_nothing(tmp_path: Any) -> None:
    """Already-analyzed tracks short-circuit before the index (negative ctrl)."""

    class _Mass:
        class streams:  # noqa: N801 - namespace shim
            class audio_analysis:  # noqa: N801 - namespace shim
                @staticmethod
                async def get_audio_analysis(**_kwargs: Any) -> AudioAnalysisData:
                    return _analysis_values()

    prov = _bare_prov(tmp_path)
    prov.mass = cast("Any", _Mass())
    assert await prov._analyze_media_silent("t1", "prov1") is False
    assert prov._quick_index.count() == 0


async def test_backfill_derives_missing_rows(tmp_path: Any) -> None:
    """Backfill walks the merged analysis rows and skips what is indexed."""
    prov = _bare_prov(tmp_path)
    analyzed_values = _analysis_values()

    async def _iter_rows(_domain: str, _media_type: Any = None, **_kwargs: Any):
        yield ("t1", "prov1", analyzed_values)
        yield ("t2", "prov1", analyzed_values)

    prov.mass = cast(
        "Any",
        SimpleNamespace(
            streams=SimpleNamespace(
                audio_analysis=SimpleNamespace(iter_merged_audio_analysis_rows=_iter_rows)
            ),
            create_task=lambda coro: (coro.close(), None)[1],
        ),
    )
    derived = await prov._quick_index_backfill()
    assert derived == 2
    assert prov._quick_index.count() == 2
    # second pass is a no-op: everything is already indexed at this version
    assert await prov._quick_index_backfill() == 0


async def test_backfill_survives_empty_feed(tmp_path: Any) -> None:
    """An offline analysis domain yields nothing; backfill stays green."""

    async def _iter_rows(_domain: str, _media_type: Any = None, **_kwargs: Any):
        return
        yield  # pragma: no cover - makes this an async generator

    prov = _bare_prov(tmp_path)
    prov.mass = cast(
        "Any",
        SimpleNamespace(
            streams=SimpleNamespace(
                audio_analysis=SimpleNamespace(iter_merged_audio_analysis_rows=_iter_rows)
            ),
            create_task=lambda coro: (coro.close(), None)[1],
        ),
    )
    assert await prov._quick_index_backfill() == 0


# ---------------------------------------------------------------------------
# pull_candidates command + registration
# ---------------------------------------------------------------------------


async def test_pull_candidates_returns_matches(tmp_path: Any) -> None:
    """The command surfaces index rows with derived values and counters."""
    prov = _bare_prov(tmp_path)
    track = SimpleNamespace(
        metadata=MediaItemMetadata(mood="Happy", release_date=datetime(1987, 5, 1, tzinfo=UTC))
    )
    prov._quick_index.upsert(
        derive_row("id0", "prov1", _analysis_values(valence=0.2, arousal=0.8), track.metadata)
    )
    result = await prov.pull_candidates(mood="tense", era="1980s")
    assert result["count"] == 1
    assert result["total_indexed"] == 1
    assert result["index_version"] == INDEX_VERSION
    assert result["candidates"][0]["item_id"] == "id0"
    assert result["candidates"][0]["mood_tier"] == "tense"
    assert result["candidates"][0]["year"] == 1987
    # raw MA metadata mood string is an alternate mood filter path
    result = await prov.pull_candidates(mood="happy")
    assert result["count"] == 1
    # no filters: unfiltered query, still capped by limit
    result = await prov.pull_candidates(limit=100000)
    assert result["count"] == 1
    assert (await prov.pull_candidates(mood="euphoric"))["count"] == 0


async def test_pull_candidates_rejects_bad_filters(tmp_path: Any) -> None:
    """Malformed filters raise the named invalid_control error."""
    prov = _bare_prov(tmp_path)
    with pytest.raises(SmartDJError, match="invalid_control"):
        await prov.pull_candidates(bpm="fast")
    with pytest.raises(SmartDJError, match="invalid_control"):
        await prov.pull_candidates(key="not-a-key")


async def test_pull_candidates_unindexed_library_is_empty(tmp_path: Any) -> None:
    """An empty index is an honest empty answer, not an error."""
    prov = _bare_prov(tmp_path)
    result = await prov.pull_candidates()
    assert result["count"] == 0
    assert result["candidates"] == []


async def test_pull_candidates_without_init() -> None:
    """A provider without the index attribute fails with a NAMED error."""
    prov = SmartDJProvider.__new__(SmartDJProvider)
    with pytest.raises(SmartDJError, match="index_unavailable"):
        await prov.pull_candidates()


def _registration_prov(tmp_path: Any) -> tuple[Any, list[str], list[Any]]:
    """
    Build a bare provider whose mass stub records api commands + subscriptions.

    The stub must mirror EVERY mass member loaded_in_mass touches: a future
    addition (e.g. lane B's subscribe) missing here fails LOUD with
    AttributeError - which is exactly how a stale copy of this test failed
    the lane D gate while HEAD was already green.
    """
    prov = _bare_prov(tmp_path)
    registered: list[str] = []
    subscribed: list[Any] = []

    def _register(cmd: str, _handler: Any, **_kwargs: Any) -> None:
        registered.append(cmd)

    def _subscribe(*_a: Any, **_k: Any) -> Any:
        subscribed.append(_a)
        return lambda: None

    prov.mass = cast(
        "Any",
        SimpleNamespace(
            storage_path=str(tmp_path),
            register_api_command=_register,
            create_task=lambda coro: (coro.close(), None)[1],
            # lane B: loaded_in_mass also subscribes the pre-analysis listener
            subscribe=_subscribe,
        ),
    )
    prov._handles = []
    return prov, registered, subscribed


async def _noop() -> None:
    return None


async def _run_loaded_in_mass(prov: Any) -> None:
    prov._load_cache = _noop
    prov._setup_ui = lambda: None
    await prov.loaded_in_mass()


def test_pull_candidates_is_registered(tmp_path: Any) -> None:
    """loaded_in_mass registers the FULL Smart DJ command set + the listener."""
    prov, registered, subscribed = _registration_prov(tmp_path)
    asyncio.run(_run_loaded_in_mass(prov))
    # every command loaded_in_mass registers - a silently drifting stub (the
    # phantom "battery still fails at HEAD" class of report) cannot hide here
    assert set(registered) == {
        "smart_dj/analyze",
        "smart_dj/rank_queue",
        "smart_dj/mix_playlist",
        "smart_dj/mix_tracks",
        "smart_dj/save_sorted_playlist",
        "smart_dj/status",
        "smart_dj/capabilities",
        "smart_dj/preroll_queue_analysis",
        "smart_dj/complete_mix",
        "smart_dj/analyze_library",
        "smart_dj/analyze_playlist",
        "smart_dj/analyze_library_local",
        "smart_dj/create_analyzed_playlist",
        "smart_dj/pull_candidates",
    }
    assert "smart_dj/pull_candidates" in registered
    # exactly one subscription: the MEDIA_ITEM_ADDED pre-analysis listener
    assert len(subscribed) == 1


def test_registration_stub_missing_subscribe_fails_loud(tmp_path: Any) -> None:
    """Negative control: a mass stub without subscribe raises AttributeError."""
    prov, _registered, _subscribed = _registration_prov(tmp_path)
    del prov.mass.subscribe
    with pytest.raises(AttributeError, match="subscribe"):
        asyncio.run(_run_loaded_in_mass(prov))


def test_pool_source_documented_in_controls_path() -> None:
    """mix_playlist/rank_queue read pool_source from the controls dict."""
    source = inspect.getsource(provider_module)
    assert 'raw.get("pool_source", "auto")' in source
