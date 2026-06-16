"""Tests for the ParametricSelfSleepHook wrapper + factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_feature_parametric_self import (
    ParametricSelfSleepHook,
    create_parametric_self_sleep_hook,
)


async def test_wrapper_delegates_to_feature():
    feature = MagicMock()
    feature.on_post_consolidation = AsyncMock(return_value={"trained": True, "promoted": True})
    hook = ParametricSelfSleepHook(feature)

    out = await hook.on_post_consolidation(MagicMock(), {"episodes_created": 2})

    feature.on_post_consolidation.assert_awaited_once_with({"episodes_created": 2})
    assert out["promoted"] is True


async def test_wrapper_swallows_training_errors():
    feature = MagicMock()
    feature.on_post_consolidation = AsyncMock(side_effect=RuntimeError("boom"))
    hook = ParametricSelfSleepHook(feature)

    out = await hook.on_post_consolidation(MagicMock(), {})

    assert out["trained"] is False
    assert "boom" in out["reason"]


async def test_pre_sleep_is_skipped():
    hook = ParametricSelfSleepHook(MagicMock())
    out = await hook.on_pre_sleep(MagicMock())
    assert out["skipped"] is True


def test_factory_returns_none_when_feature_absent():
    agent = MagicMock()
    agent.get_feature = MagicMock(return_value=None)
    assert create_parametric_self_sleep_hook(agent) is None


def test_factory_returns_hook_when_feature_present():
    agent = MagicMock()
    feature = MagicMock()
    agent.get_feature = MagicMock(side_effect=lambda n: feature if n in ("parametric_self", "ParametricSelfFeature") else None)
    hook = create_parametric_self_sleep_hook(agent)
    assert isinstance(hook, ParametricSelfSleepHook)
    assert hook.feature is feature
