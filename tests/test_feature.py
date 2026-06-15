"""P0 scaffold tests for ParametricSelfFeature.

Mirrors the kestrel-feature-reflection test style: import the feature, assert
it registers through the ``kestrel_sovereign.features`` entry point, and drive
the scaffold surfaces (status tool + sleep hook) to confirm they load and
return the expected shapes before the real trainer arrives in P1+.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_feature_parametric_self import ParametricSelfFeature


def _feature() -> ParametricSelfFeature:
    """Construct the feature with a mock agent (Feature.__init__ takes agent)."""
    return ParametricSelfFeature(agent=MagicMock())


def test_entry_point_registered():
    """The feature is discoverable through the features entry point group."""
    eps = entry_points(group="kestrel_sovereign.features")
    assert any(ep.name == "ParametricSelfFeature" for ep in eps)


def test_tool_description_is_nonempty():
    feature = _feature()
    assert isinstance(feature.tool_description, str)
    assert feature.tool_description.strip()


async def test_status_tool_reports_scaffold_phase():
    """The status tool returns a successful ToolResult describing P0 state."""
    feature = _feature()
    await feature.initialize()

    result = await feature.parametric_self_status()

    assert result.status == ToolResultStatus.OK
    assert result.data["phase"].startswith("P0")
    assert result.data["trainer_available"] is False
    assert result.data["in_loop"] is False


async def test_post_consolidation_hook_is_a_noop_in_p0():
    """The sleep hook loads and declines to train until P2."""
    feature = _feature()
    await feature.initialize()

    result = await feature.on_post_consolidation({"episodes_created": 3})

    assert result["trained"] is False
    assert result["promoted"] is False
    assert "P2" in result["reason"]
