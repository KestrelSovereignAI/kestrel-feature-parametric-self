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


def test_command_prefixes_do_not_shadow_each_other():
    """No tool's command prefix may be a prefix of another's, or the dispatcher
    (first startswith-match in get_skill_for_command) would route to the wrong
    tool — e.g. a bare `!parametric-self` would swallow `!parametric-self train-now`."""
    feature = ParametricSelfFeature(agent=MagicMock())
    prefixes = [t.schema.command_prefix for t in feature.get_tools() if t.schema.command_prefix]
    assert prefixes, "expected command-prefixed tools"
    for a in prefixes:
        for b in prefixes:
            if a is not b and a != b:
                assert not b.startswith(a + " ") and b != a, (
                    f"command prefix {a!r} shadows {b!r}"
                )


def test_each_subcommand_resolves_to_its_own_tool():
    """Every `!parametric-self <verb>` resolves to the matching tool (not status)."""
    feature = ParametricSelfFeature(agent=MagicMock())
    cases = {
        "!parametric-self status": "parametric-self-status",
        "!parametric-self history": "parametric-self-history",
        "!parametric-self adapters": "parametric-self-adapters",
        "!parametric-self train-now": "parametric-self-train-now",
        "!parametric-self set-enabled true": "parametric-self-set-enabled",
        "!parametric-self rollback": "parametric-self-rollback",
    }
    for command, expected in cases.items():
        assert feature.get_skill_for_command(command) == expected, command
