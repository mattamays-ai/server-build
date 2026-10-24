"""
Regression pins for the Oct 24 full-program review fixes (0.0.29).

Each test here pins a finding that survived adversarial verification:
the tombstone read path, the analysis success-polls, the mix_tracks `uris`
alias, the rank_queue fail-soft plan contract, the stems veto override,
the preroll hard timeout, and the index version purge.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from music_assistant.controllers.streams.smart_fades.planner.assembly import (
    PlanAssembler,
)
from music_assistant.models.audio_analysis import AudioAnalysisData
from music_assistant.providers.smart_dj import provider as provider_module
from music_assistant.providers.smart_dj.provider import (
    STEMS_VETO_DISABLED,
    STEMS_VETO_OUTGOING_VOCAL_DOMINANCE,
    SmartDJProvider,
    _stems_veto_reason,
)
from music_assistant.providers.smart_dj.quick_index import INDEX_VERSION, QuickIndex

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _bare_prov(tmp_path: Path) -> SmartDJProvider:
    """Minimal bare provider with a temp quick index (mirrors test_quick_index)."""
    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov.logger = logging.getLogger("smart_dj_review_test")
    prov._quick_index = QuickIndex(tmp_path / "qi.sqlite3")
    prov._quick_index.open()
    return cast("SmartDJProvider", prov)


# --- tombstone read path -----------------------------------------------------


async def test_silent_analyzer_treats_a_tombstone_as_not_analyzed() -> None:
    """The skip gate must not treat an acoustid tombstone as already-analyzed."""

    async def _tombstoned(**_kwargs: Any) -> Any:
        return AudioAnalysisData(extra_data={"retry_after": 1234})

    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov.logger = logging.getLogger("smart_dj_review_test")
    prov.mass = cast(
        "Any",
        SimpleNamespace(
            get_provider=lambda *_a, **_k: None,
            streams=SimpleNamespace(
                audio_analysis=SimpleNamespace(get_audio_analysis=_tombstoned)
            ),
        ),
    )
    # the gate runs BEFORE the provider lookup: a tombstoned existing analysis
    # must fall through (return False for "cannot analyze"), never short-circuit
    # as already-analyzed - that was the bug that permanently froze 665 tracks
    assert await prov._analyze_media_silent("x1", "prov1") is False


async def test_analysis_ma_branch_refuses_a_tombstone() -> None:
    """The planner read path must not surface a phantom analyzed dict."""

    async def _tombstoned(**_kwargs: Any) -> Any:
        return AudioAnalysisData(extra_data={"retry_after": 1234})

    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov.logger = logging.getLogger("smart_dj_review_test")
    prov._cache = {}
    prov.mass = cast(
        "Any",
        SimpleNamespace(
            get_provider=lambda *_a, **_k: None,
            streams=SimpleNamespace(
                audio_analysis=SimpleNamespace(get_audio_analysis=_tombstoned)
            ),
        ),
    )
    result = await prov._analysis("x1", "prov1", None)
    # a tombstone is NOT an analysis: no phantom music_assistant dict
    assert result is None or result.get("source") != "music_assistant"


# --- index version purge ------------------------------------------------------


def test_index_open_purges_rows_from_older_versions(tmp_path: Path) -> None:
    """Re-opening after the version bump deletes pre-bump rows (shells)."""
    index = QuickIndex(tmp_path / "qi.sqlite3")
    index.open()
    stale = dict.fromkeys(
        (
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
        ),
        None,
    )
    stale.update({"item_id": "t1", "provider_instance": "prov1", "updated_at": 1.0})
    stale["analyzer_version"] = INDEX_VERSION - 1
    conn = sqlite3.connect(tmp_path / "qi.sqlite3")
    conn.execute(
        f"INSERT INTO track_index ({', '.join(stale)}) "
        f"VALUES ({', '.join('?' for _ in stale)})",
        tuple(stale.values()),
    )
    conn.commit()
    conn.close()
    # a fresh open() must purge the stale-version row
    index = QuickIndex(tmp_path / "qi.sqlite3")
    index.open()
    assert index.count() == 0


# --- mix_tracks uris alias -----------------------------------------------------


async def test_mix_tracks_accepts_uris_alias(tmp_path: Path) -> None:
    """The FE's shipped payload key (`uris`) must bind, not crash."""
    prov = _bare_prov(tmp_path)
    captured: dict[str, Any] = {}

    async def _resolve(_uris: list[str]) -> tuple[list[Any], list[str]]:
        return [SimpleNamespace(name="t1"), SimpleNamespace(name="t2")], []

    async def _mix_resolved(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    prov._resolve_track_uris = _resolve  # type: ignore[method-assign]
    prov._mix_resolved = _mix_resolved  # type: ignore[method-assign]
    result = await prov.mix_tracks("q1", uris=["library://track/1", "library://track/2"])
    assert result == {"ok": True}
    assert captured["response_base"]["track_uris"] == 2


# --- rank_queue fail-soft plan contract ---------------------------------------


async def test_rank_queue_returns_plan_failed_contract(tmp_path: Path) -> None:
    """A HardRequirementsImpossible becomes the FE's plan_failed shape, not a raise."""
    prov = _bare_prov(tmp_path)
    snapshot = {
        "tracks": [
            {"queue_item_id": "q0", "name": "cur"},
            {"queue_item_id": "q1", "name": "a"},
            {"queue_item_id": "q2", "name": "b"},
        ],
        "current": {"queue_item_id": "q0", "name": "cur"},
        "queue_item_count": 3,
    }

    async def _analyze(*_a: Any, **_k: Any) -> dict[str, Any]:
        return dict(snapshot)

    prov.analyze = _analyze  # type: ignore[method-assign]

    real_beam = provider_module.beam_optimize

    def _impossible(*_a: Any, **_k: Any) -> None:
        raise provider_module.HardRequirementsImpossible(
            "Hard requirements are impossible with the current queue",
            [
                {
                    "rule": "bpm_delta",
                    "actual": 190.0,
                    "limit": 150.0,
                    "unit": "bpm",
                    "suggested_setting": "bpm_tolerance",
                    "suggested_value": 0.12,
                }
            ],
        )

    provider_module.beam_optimize = _impossible  # type: ignore[assignment]
    try:
        result = await prov.rank_queue("q1")
    finally:
        provider_module.beam_optimize = real_beam  # type: ignore[assignment]
    assert result["plan_failed"] is True
    rows = result["hard_violations"]
    assert rows
    assert rows[0]["rule"] == "bpm_delta"
    assert rows[0]["suggestion"]["setting"] == "bpm_tolerance"
    assert result["applied"] is False


# --- stems veto override -------------------------------------------------------


def test_veto_override_forces_the_duck_past_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override escape hatch beats the threshold vetoes, never the toggle."""
    monkeypatch.setattr(
        provider_module.stems_ducking, "available", lambda: True
    )
    dense = {"vocal_density": 0.9}
    # without override: a vocal-dense outgoing is vetoed
    assert _stems_veto_reason(dense, True) == STEMS_VETO_OUTGOING_VOCAL_DOMINANCE
    # with override: the duck runs
    assert _stems_veto_reason(dense, True, veto_override=True) is None
    # the toggle stays master: an override never forces stems ON
    assert (
        _stems_veto_reason(dense, False, veto_override=True) == STEMS_VETO_DISABLED
    )


# --- preroll hard timeout ------------------------------------------------------


async def test_preroll_timeout_is_recorded_as_a_failure(tmp_path: Path) -> None:
    """A hung preroll fetch burns the 45s budget once, then is an honest failure."""
    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov.logger = logging.getLogger("smart_dj_review_test")
    prov._cache = {}
    prov._quick_index = QuickIndex(tmp_path / "qi.sqlite3")
    prov._quick_index.open()

    class _Queue:
        current_index = 0

    class _Current:
        queue_item_id = "cur0"
        name = "current"

    class _Item:
        queue_item_id = "up1"
        name = "upcoming"
        media_type = provider_module.MediaType.TRACK

    def _get(_queue_id: str) -> Any:
        return _Queue()

    async def _hung(_item: Any) -> bool:
        await asyncio.sleep(300)
        return True

    prov.mass = cast(
        "Any",
        SimpleNamespace(
            player_queues=SimpleNamespace(get=_get),
            streams=SimpleNamespace(audio_analysis=SimpleNamespace(providers=[object()])),
        ),
    )
    prov._read_full_queue = lambda _qid: [_Current(), _Item()]  # type: ignore[method-assign]
    prov._preroll_track_analysis = _hung  # type: ignore[method-assign]
    real_timeout = provider_module.ANALYSIS_ATTEMPT_TIMEOUT
    provider_module.ANALYSIS_ATTEMPT_TIMEOUT = 0.05  # type: ignore[assignment]
    try:
        result = await prov.preroll_queue_analysis("q1", 5)
    finally:
        provider_module.ANALYSIS_ATTEMPT_TIMEOUT = real_timeout
    assert result["prerolled"] == []
    assert "timeout after" in result["failed"]["up1"]


# --- audio path pin (fallback gain match entry point) --------------------------


def test_plan_assembler_gain_match_exists_for_factory_plans() -> None:
    """The fallback/handoff arm's seam-correction entry point is present."""
    assert hasattr(PlanAssembler, "gain_match")
