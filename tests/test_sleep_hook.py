"""Tests for the ParametricSelfSleepHook wrapper + factory."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sovereign.agent.sleep import SleepHookStatus, SleepMixin

from kestrel_feature_parametric_self import (
    ParametricSelfFeature,
    ParametricSelfSleepHook,
    create_parametric_self_sleep_hook,
)
from kestrel_feature_parametric_self.cycle import TrainingShutdownIncomplete


async def test_wrapper_delegates_to_feature():
    feature = MagicMock()
    feature.on_post_consolidation = AsyncMock(return_value={"trained": True, "promoted": True})
    hook = ParametricSelfSleepHook(feature)

    out = await hook.on_post_consolidation(MagicMock(), {"episodes_created": 2})

    feature.on_post_consolidation.assert_awaited_once_with({"episodes_created": 2})
    assert out["promoted"] is True
    assert out["success"] is True
    assert out["skipped"] is False


async def test_wrapper_swallows_training_errors():
    feature = MagicMock()
    feature.on_post_consolidation = AsyncMock(side_effect=RuntimeError("boom"))
    hook = ParametricSelfSleepHook(feature)

    out = await hook.on_post_consolidation(MagicMock(), {})

    assert out["trained"] is False
    assert "boom" in out["reason"]
    assert out["success"] is False
    assert SleepMixin._hook_outcome(out)[0] is SleepHookStatus.FAILED


async def test_governed_corpus_unavailability_is_a_visible_sleep_failure():
    feature = MagicMock()
    feature.on_post_consolidation = AsyncMock(return_value={
        "trained": False,
        "promoted": False,
        "reason": "governed corpus unavailable or semantic maintenance incomplete",
    })
    hook = ParametricSelfSleepHook(feature)

    out = await hook.on_post_consolidation(MagicMock(), {})

    assert out["success"] is False
    assert out["skipped"] is False
    assert SleepMixin._hook_outcome(out)[0] is SleepHookStatus.FAILED


@pytest.mark.parametrize(
    "reason",
    ["trainer unavailable on this host", "empty corpus — no grounded reflections to train on"],
)
async def test_expected_local_training_noops_are_structured_sleep_skips(reason):
    feature = MagicMock()
    feature.on_post_consolidation = AsyncMock(return_value={
        "trained": False, "promoted": False, "reason": reason,
    })
    hook = ParametricSelfSleepHook(feature)

    out = await hook.on_post_consolidation(MagicMock(), {})

    assert out["success"] is True
    assert out["skipped"] is True
    assert SleepMixin._hook_outcome(out)[0] is SleepHookStatus.SKIPPED


async def test_enabled_unsupported_trainer_skips_before_governed_policy_lookup():
    """A real feature hook on an unsupported host must not fail for no policy."""
    agent = MagicMock()
    agent.is_test_instance = False
    agent.storage = None
    agent.storage_path = None
    agent.parametric_self_governed_corpus_policy = None
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    feature._training_enabled = True
    feature._adapter.is_available = lambda: False

    out = await ParametricSelfSleepHook(feature).on_post_consolidation(agent, {})

    assert out["reason"] == "trainer unavailable on this host"
    assert SleepMixin._hook_outcome(out)[0] is SleepHookStatus.SKIPPED


async def test_enabled_supported_trainer_without_policy_remains_a_sleep_failure(tmp_path):
    agent = MagicMock()
    agent.is_test_instance = False
    agent.storage = None
    agent.storage_path = None
    agent.parametric_self_work_dir = str(tmp_path / "work")
    agent.parametric_self_governed_corpus_policy = None
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    feature._training_enabled = True
    feature._adapter.is_available = lambda: True

    out = await ParametricSelfSleepHook(feature).on_post_consolidation(agent, {})

    assert out["reason"] == "governed corpus policy is not configured"
    assert SleepMixin._hook_outcome(out)[0] is SleepHookStatus.FAILED


async def test_actual_hook_marks_feature_disable_interruption_as_skipped():
    """The expected feature-disable path must not fail the core sleep cycle."""
    agent = MagicMock()
    agent.is_test_instance = False
    agent.storage = None
    agent.storage_path = None
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    feature._training_enabled = True
    feature._adapter.is_available = lambda: True
    started = asyncio.Event()

    async def _slow_cycle(*, trigger):
        started.set()
        await asyncio.Event().wait()

    async def _confirmed_cancel_all():
        return SimpleNamespace(all_stopped=True)

    feature._run_training_cycle_locked = _slow_cycle
    feature._adapter.cancel_all = _confirmed_cancel_all
    hook_task = asyncio.create_task(
        ParametricSelfSleepHook(feature).on_post_consolidation(agent, {})
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    await feature.on_disable()
    out = await hook_task
    assert out["success"] is True
    assert out["skipped"] is True
    assert SleepMixin._hook_outcome(out)[0] is SleepHookStatus.SKIPPED


async def test_actual_hook_keeps_unconfirmed_trainer_shutdown_failed():
    """Unsafe trainer shutdown remains a visible failed sleep dependency."""
    agent = MagicMock()
    agent.is_test_instance = False
    agent.storage = None
    agent.storage_path = None
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    feature._training_enabled = True
    feature._adapter.is_available = lambda: True
    started = asyncio.Event()

    async def _unconfirmed_cycle(*, trigger):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise TrainingShutdownIncomplete("trainer stop could not be confirmed")

    async def _unconfirmed_cancel_all():
        return SimpleNamespace(all_stopped=False)

    feature._run_training_cycle_locked = _unconfirmed_cycle
    feature._adapter.cancel_all = _unconfirmed_cancel_all
    hook_task = asyncio.create_task(
        ParametricSelfSleepHook(feature).on_post_consolidation(agent, {})
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    await feature.on_disable()
    out = await hook_task
    assert out["success"] is False
    assert out["skipped"] is False
    assert SleepMixin._hook_outcome(out)[0] is SleepHookStatus.FAILED


async def test_pre_sleep_is_skipped():
    hook = ParametricSelfSleepHook(MagicMock())
    out = await hook.on_pre_sleep(MagicMock())
    assert out["skipped"] is True
    assert SleepMixin._hook_outcome(out)[0] is SleepHookStatus.SKIPPED


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
