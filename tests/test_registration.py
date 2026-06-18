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


def test_command_prefixes_are_single_token_and_non_nesting():
    """Each command prefix must be a single token (no spaces, so the dispatcher
    routes by exact first-token match and no-arg tools receive no stray arg0) and
    no prefix may be a textual prefix of another (so the startswith fallback in
    get_skill_for_command stays unambiguous too)."""
    feature = ParametricSelfFeature(agent=MagicMock())
    prefixes = [t.schema.command_prefix for t in feature.get_tools() if t.schema.command_prefix]
    assert prefixes, "expected command-prefixed tools"
    for p in prefixes:
        assert " " not in p, f"command prefix {p!r} must be a single token"
    for a in prefixes:
        for b in prefixes:
            if a != b:
                assert not b.startswith(a), f"command prefix {a!r} shadows {b!r}"


def test_feature_exposes_get_tool_by_name_for_rich_arg_parsing():
    """The handler must expose _get_tool_by_name so the host uses the tool's
    prefix-aware, type-coercing parse_command_args (binding positional args)
    instead of the naive splitter that emits an arg0 our methods reject."""
    feature = ParametricSelfFeature(agent=MagicMock())
    tool = feature._get_tool_by_name("parametric-self-set-enabled")
    assert tool is not None and hasattr(tool, "parse_command_args")
    # Positional bool binds to the named param and is coerced.
    assert tool.parse_command_args("!parametric-self-enable false") == {"enabled": False}
    assert tool.parse_command_args("!parametric-self-enable true") == {"enabled": True}
    # No-arg command yields no stray positional.
    status = feature._get_tool_by_name("parametric-self-status")
    assert status.parse_command_args("!parametric-self-status") == {}


def test_each_command_resolves_to_its_own_tool():
    """Every `!parametric-self-<verb>` resolves to the matching tool (not status)."""
    feature = ParametricSelfFeature(agent=MagicMock())
    cases = {
        "!parametric-self-status": "parametric-self-status",
        "!parametric-self-history": "parametric-self-history",
        "!parametric-self-adapters": "parametric-self-adapters",
        "!parametric-self-progress": "parametric-self-progress",
        "!parametric-self-train": "parametric-self-train-now",
        "!parametric-self-enable enabled=false": "parametric-self-set-enabled",
        "!parametric-self-rollback adapter_id=abc": "parametric-self-rollback",
        "!parametric-self-adopt adapter_id=abc": "parametric-self-adopt",
    }
    for command, expected in cases.items():
        assert feature.get_skill_for_command(command) == expected, command
