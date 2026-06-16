"""Parametric-self registers its sleep hook on the core sleep_hooks list (#1784)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kestrel_feature_parametric_self import ParametricSelfFeature


async def test_post_all_features_loaded_appends_to_sleep_hooks():
    agent = MagicMock()
    agent.sleep_hooks = []
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    agent.get_feature = MagicMock(
        side_effect=lambda n: feature if n in ("parametric_self", "ParametricSelfFeature") else None
    )

    await feature.post_all_features_loaded(agent)

    assert len(agent.sleep_hooks) == 1
    assert agent.sleep_hooks[0] is feature._sleep_hook
    assert hasattr(agent.sleep_hooks[0], "on_post_consolidation")


async def test_on_disable_unregisters_and_is_idempotent():
    agent = MagicMock()
    agent.sleep_hooks = []
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    agent.get_feature = MagicMock(
        side_effect=lambda n: feature if n in ("parametric_self", "ParametricSelfFeature") else None
    )

    await feature.post_all_features_loaded(agent)
    # re-registering must not duplicate
    await feature.post_all_features_loaded(agent)
    assert len(agent.sleep_hooks) == 1

    await feature.on_disable()
    assert agent.sleep_hooks == []  # hook removed on disable


async def test_registration_initializes_list_when_absent():
    agent = MagicMock()
    agent.sleep_hooks = None
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    agent.get_feature = MagicMock(
        side_effect=lambda n: feature if n in ("parametric_self", "ParametricSelfFeature") else None
    )

    await feature.post_all_features_loaded(agent)

    assert isinstance(agent.sleep_hooks, list)
    assert len(agent.sleep_hooks) == 1
