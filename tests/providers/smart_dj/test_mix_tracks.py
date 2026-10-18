"""Tests for the 0.0.27 smart_dj/mix_tracks command (track-uri mixing)."""

from __future__ import annotations

import asyncio
import inspect
import logging
from types import SimpleNamespace
from typing import Any

from music_assistant_models.enums import MediaType

from music_assistant.providers.smart_dj.provider import SmartDJProvider


def _bare_provider() -> Any:
    """Return a bare provider instance (no framework wiring)."""
    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov.logger = logging.getLogger("smart-dj-test")
    return prov


def _fake_mass(handler: Any) -> Any:
    """Return a fake mass host whose get_item_by_uri calls ``handler``."""
    music = SimpleNamespace(get_item_by_uri=handler)
    return SimpleNamespace(music=music)


def _track(item_id: str) -> Any:
    """Return a minimal Track-shaped item."""
    return SimpleNamespace(media_type=MediaType.TRACK, item_id=item_id, name=f"track-{item_id}")


def test_resolve_track_uris_found_and_not_found() -> None:
    """Resolvable track URIs resolve; failures and non-tracks are reported."""

    async def handler(uri: str) -> Any:
        if uri == "bad://track/boom":
            raise RuntimeError(f"provider not found: {uri}")
        if uri == "library://playlist/1":
            return SimpleNamespace(media_type=MediaType.PLAYLIST, item_id="1")
        return _track(uri.rsplit("/", 1)[-1])

    prov = _bare_provider()
    prov.mass = _fake_mass(handler)
    resolved, unresolved = asyncio.run(
        prov._resolve_track_uris(["library://track/a", "bad://track/boom", "library://playlist/1"])
    )
    assert [t.item_id for t in resolved] == ["a"]
    assert unresolved == ["bad://track/boom", "library://playlist/1"]


def test_resolve_track_uris_empty_input() -> None:
    """An empty URI list resolves nothing and reports nothing."""
    prov = _bare_provider()
    prov.mass = _fake_mass(lambda _uri: _track("x"))
    resolved, unresolved = asyncio.run(prov._resolve_track_uris([]))
    assert resolved == []
    assert unresolved == []


def test_mix_tracks_not_enough_resolvable_tracks_raises() -> None:
    """Fewer than 2 resolvable tracks must fail loudly, not half-mix."""
    import pytest  # noqa: PLC0415 - pytest is not a sandbox dependency

    async def handler(uri: str) -> Any:
        raise RuntimeError(f"no provider for {uri}")

    prov = _bare_provider()
    prov.mass = _fake_mass(handler)
    with pytest.raises(RuntimeError, match="at least 2 resolvable tracks"):
        asyncio.run(prov.mix_tracks("q1", ["bad://a", "bad://b"]))


def test_mix_tracks_threads_controls_and_response_base() -> None:
    """mix_tracks parses controls and hands the pipeline the right base keys."""
    prov = _bare_provider()
    captured: dict[str, Any] = {}

    async def fake_resolve(_uris: list[str]) -> tuple[list[Any], list[str]]:
        return [_track("a"), _track("b")], ["junk://x"]

    async def fake_pipeline(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"shared": True}

    prov._resolve_track_uris = fake_resolve
    prov._mix_resolved = fake_pipeline
    response = asyncio.run(
        prov.mix_tracks(
            "q1",
            ["library://track/a", "library://track/b", "junk://x"],
            controls={"energy_arc": True, "transition_bars": 16},
        )
    )
    assert response == {"shared": True}
    assert captured["queue_id"] == "q1"
    assert captured["analysis_provider"] == "auto"
    assert captured["pool_source"] == "auto"
    assert captured["response_base"] == {"track_uris": 3, "unresolved_uris": ["junk://x"]}
    control = captured["control"]
    assert control.energy_arc is True
    assert control.transition_bars == 16


def test_mix_tracks_registration_is_member_only() -> None:
    """mix_tracks can replace a whole queue: never guest-safe."""
    source = inspect.getsource(SmartDJProvider.loaded_in_mass)
    assert '("smart_dj/mix_tracks", self.mix_tracks, Scope.LIBRARY_WRITE)' in source


def test_mix_playlist_contract_untouched() -> None:
    """mix_playlist's signature keeps its playlist-only shape."""
    sig = inspect.signature(SmartDJProvider.mix_playlist)
    assert list(sig.parameters)[:4] == ["self", "queue_id", "playlist_provider", "playlist_item_id"]
    sig_tracks = inspect.signature(SmartDJProvider.mix_tracks)
    assert list(sig_tracks.parameters)[:3] == ["self", "queue_id", "track_uris"]
