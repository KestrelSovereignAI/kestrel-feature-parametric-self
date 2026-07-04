"""Feature-level tests for ParametricSelfFeature.

Import + entry-point registration, the status tool shape, and the default-OFF
training gate. Avoids asserting platform-specific trainer availability (this
host may or may not have MLX) — only behavior and types.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_feature_parametric_self import ParametricSelfFeature


def _feature() -> ParametricSelfFeature:
    return ParametricSelfFeature(agent=MagicMock())


def test_entry_point_registered():
    eps = entry_points(group="kestrel_sovereign.features")
    assert any(ep.name == "ParametricSelfFeature" for ep in eps)


def test_tool_description_is_nonempty():
    feature = _feature()
    assert isinstance(feature.tool_description, str)
    assert feature.tool_description.strip()


async def test_status_tool_reports_state():
    feature = _feature()
    await feature.initialize()

    result = await feature.parametric_self_status()

    assert result.status == ToolResultStatus.OK
    assert result.data["training_enabled"] is False  # default OFF
    assert isinstance(result.data["trainer_available"], bool)
    assert result.data["served_adapter"] is None


async def test_training_disabled_by_default_is_a_noop():
    feature = _feature()
    await feature.initialize()

    result = await feature.on_post_consolidation({"episodes_created": 3})

    assert result["trained"] is False
    assert result["promoted"] is False
    assert "disabled" in result["reason"]


async def test_resolve_paths_uses_storage_path(tmp_path):
    """The cognition DB + work dir derive from agent.storage_path (not data_dir)."""
    agent = MagicMock()
    db = tmp_path / "kestrel_prime.db"
    agent.storage_path = str(db)
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()

    db_path, work_dir = feature._resolve_paths()
    assert db_path == str(db)
    assert work_dir == str(tmp_path / "parametric_self")


async def test_resolve_paths_none_when_no_storage_path():
    agent = MagicMock()
    agent.storage_path = None
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    assert feature._resolve_paths() == (None, None)


async def test_set_config_can_enable_training():
    feature = _feature()
    await feature.initialize()
    assert feature._training_enabled is False

    await feature.set_config({"enable_nightly_training": True})
    assert feature._training_enabled is True

    cfg = await feature.get_config()
    assert cfg["enable_nightly_training"] is True


@pytest.mark.asyncio
async def test_training_skipped_when_privacy_hides_user_content(monkeypatch):
    """F377: when the privacy mode forbids durably retaining user content, the
    training cycle must be skipped (not baked into adapter weights)."""
    agent = MagicMock()
    agent.is_test_instance = False  # sovereign-class → passes the Incubator gate
    feature = ParametricSelfFeature(agent=agent)
    monkeypatch.setattr(feature, "_hides_persisted_user_content", lambda: True)

    ran = []

    async def _locked(*, trigger):
        ran.append(trigger)
        return {"trained": True, "promoted": False}

    monkeypatch.setattr(feature, "_run_training_cycle_locked", _locked)

    result = await feature._run_training_cycle(trigger="manual")

    assert result["trained"] is False
    assert "privacy mode" in result["reason"]
    assert ran == []  # the cycle body never ran


@pytest.mark.asyncio
async def test_training_proceeds_when_privacy_allows(monkeypatch):
    agent = MagicMock()
    agent.is_test_instance = False
    feature = ParametricSelfFeature(agent=agent)
    feature._cycle_in_flight = False
    monkeypatch.setattr(feature, "_hides_persisted_user_content", lambda: False)

    ran = []

    async def _locked(*, trigger):
        ran.append(trigger)
        return {"trained": True, "promoted": False}

    monkeypatch.setattr(feature, "_run_training_cycle_locked", _locked)

    result = await feature._run_training_cycle(trigger="manual")

    assert ran == ["manual"]  # the cycle body ran
    assert result["trained"] is True


@pytest.mark.asyncio
async def test_manual_train_now_refused_under_privacy(monkeypatch):
    """F377 (codex P1): the detached manual `parametric_self_train_now` tool
    must also refuse when privacy hides persisted user content — it bypasses
    the nightly `_run_training_cycle` gate."""
    agent = MagicMock()
    agent.is_test_instance = False
    feature = ParametricSelfFeature(agent=agent)
    feature._cycle_in_flight = False
    feature._training_task = None
    monkeypatch.setattr(feature, "_hides_persisted_user_content", lambda: True)

    started = []
    monkeypatch.setattr(
        feature, "_begin_active_run",
        lambda **kw: started.append(kw) or {"run_id": "r", "adapter_id": "a", "adapter_path": "p"},
    )

    result = await feature.parametric_self_train_now()

    assert result.status is ToolResultStatus.ERROR
    assert "privacy mode" in (result.error or "")
    assert started == []  # never started / created a run record
    assert feature._cycle_in_flight is False
