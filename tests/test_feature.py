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
