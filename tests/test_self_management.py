"""Agent introspection + sovereign-class self-management tools (epic #10).

Covers the two view tools (history, adapters), the three mutation tools
(train_now, set_enabled, rollback), and — critically — the Incubator-Principle
gate: governed/test instances must be refused, sovereign-class agents allowed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_feature_parametric_self import ParametricSelfFeature


class _FakeStorage:
    """In-memory stand-in for the agent graph store (add_node/get_node)."""

    def __init__(self):
        self.nodes = {}

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)


def _agent(storage=None, *, is_test_instance=False, storage_path=None):
    agent = MagicMock()
    agent.storage = storage
    agent.sleep_hooks = []
    # Real agents expose a bool property; a bare MagicMock would be truthy and
    # silently flip the sovereign-class gate, so set it explicitly.
    agent.is_test_instance = is_test_instance
    agent.storage_path = storage_path
    agent.agent_id = "test-agent"
    return agent


async def _feature(storage=None, *, is_test_instance=False, storage_path=None):
    f = ParametricSelfFeature(agent=_agent(
        storage, is_test_instance=is_test_instance, storage_path=storage_path,
    ))
    await f.initialize()
    return f


# ----------------------------------------------------------------------
# View tools (ungated)
# ----------------------------------------------------------------------

async def test_history_empty_is_ok():
    f = await _feature(_FakeStorage())
    result = await f.parametric_self_history()
    assert result.status == ToolResultStatus.OK
    assert result.data["runs"] == []


async def test_history_lists_runs_most_recent_first():
    f = await _feature(_FakeStorage())
    await f._append_run_history({"timestamp": "2026-06-17T01:00:00+00:00", "trigger": "nightly",
                                 "trained": True, "promoted": True, "val_loss": 2.6,
                                 "corpus_train": 400, "reason": "ok", "adapter_path": "/c/a"})
    await f._append_run_history({"timestamp": "2026-06-18T01:00:00+00:00", "trigger": "manual",
                                 "trained": True, "promoted": False, "val_loss": 3.4,
                                 "corpus_train": 410, "reason": "regressed", "adapter_path": None})
    result = await f.parametric_self_history()
    assert result.status == ToolResultStatus.OK
    runs = result.data["runs"]
    assert len(runs) == 2
    assert runs[0]["trigger"] == "manual"  # most recent first
    assert runs[1]["trigger"] == "nightly"


async def test_adapters_lists_candidates_and_marks_served(tmp_path):
    work = tmp_path / "parametric_self"
    cands = work / "candidates"
    for name, loss in (("aaa111", "Val loss 2.601"), ("bbb222", "Val loss 3.100")):
        d = cands / name
        d.mkdir(parents=True)
        (d / "train.log").write_text(f"Iter 400: {loss}\n")
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._active_adapter_path = str(cands / "aaa111")

    result = await f.parametric_self_adapters()
    assert result.status == ToolResultStatus.OK
    by_id = {a["adapter_id"]: a for a in result.data["adapters"]}
    assert by_id["aaa111"]["served"] is True
    assert by_id["bbb222"]["served"] is False
    assert by_id["aaa111"]["val_loss"] == pytest.approx(2.601)


# ----------------------------------------------------------------------
# Incubator-Principle gate
# ----------------------------------------------------------------------

async def test_mutation_tools_refused_for_governed_agent():
    f = await _feature(_FakeStorage(), is_test_instance=True)
    for result in (
        await f.parametric_self_train_now(),
        await f.parametric_self_set_enabled(True),
        await f.parametric_self_rollback(),
    ):
        assert result.status == ToolResultStatus.ERROR
        assert "Incubator Principle" in (result.error or "")
    # The gate must not have flipped state.
    assert f._training_enabled is False


async def test_view_tools_allowed_for_governed_agent():
    f = await _feature(_FakeStorage(), is_test_instance=True)
    assert (await f.parametric_self_history()).status == ToolResultStatus.OK
    assert (await f.parametric_self_adapters()).status == ToolResultStatus.OK


# ----------------------------------------------------------------------
# Mutation tools (sovereign-class)
# ----------------------------------------------------------------------

async def test_set_enabled_toggles_and_persists():
    storage = _FakeStorage()
    f = await _feature(storage)
    result = await f.parametric_self_set_enabled(True)
    assert result.status == ToolResultStatus.OK
    assert f._training_enabled is True
    assert f._config_node_id() in storage.nodes  # persisted

    # A fresh instance restores the enablement.
    f2 = ParametricSelfFeature(agent=f.agent)
    await f2.initialize()
    await f2._restore_persisted_config()
    assert f2._training_enabled is True


async def test_train_now_unavailable_trainer_fails():
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: False
    result = await f.parametric_self_train_now()
    assert result.status == ToolResultStatus.ERROR
    assert "unavailable" in (result.error or "")


async def test_train_now_starts_detached_run():
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    ran = asyncio.Event()

    async def _fake_cycle(*, trigger):
        ran.set()
        return {"trained": True, "promoted": False}

    f._run_training_cycle = _fake_cycle

    result = await f.parametric_self_train_now()
    assert result.status == ToolResultStatus.OK
    assert result.data["started"] is True
    await asyncio.wait_for(ran.wait(), timeout=2)
    await f._training_task  # detached task completed cleanly

    # A second call while a run is in flight is refused.
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_cycle(*, trigger):
        started.set()
        await release.wait()
        return {}

    f._run_training_cycle = _slow_cycle
    first = await f.parametric_self_train_now()
    assert first.status == ToolResultStatus.OK
    await asyncio.wait_for(started.wait(), timeout=2)
    busy = await f.parametric_self_train_now()
    assert busy.status == ToolResultStatus.ERROR
    assert "in progress" in (busy.error or "")
    release.set()
    await f._training_task


async def test_rollback_default_to_previous_promoted(tmp_path):
    work = tmp_path / "parametric_self"
    cands = work / "candidates"
    old = cands / "old111"
    new = cands / "new222"
    for d, loss in ((old, "2.900"), (new, "2.600")):
        d.mkdir(parents=True)
        (d / "train.log").write_text(f"Val loss {loss}\n")

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    # History records both promotions; new is currently served.
    await f._append_run_history({"promoted": True, "adapter_path": str(old), "trigger": "nightly"})
    await f._append_run_history({"promoted": True, "adapter_path": str(new), "trigger": "nightly"})
    f._active_adapter_path = str(new)

    result = await f.parametric_self_rollback()
    assert result.status == ToolResultStatus.OK
    assert f._active_adapter_path == str(old)
    assert f._last_val_loss == pytest.approx(2.900)
    # Rollback is recorded in history.
    runs = await f._load_run_history()
    assert runs[-1]["trigger"] == "rollback"


async def test_rollback_explicit_adapter_id(tmp_path):
    work = tmp_path / "parametric_self"
    cands = work / "candidates"
    target = cands / "pick99"
    target.mkdir(parents=True)
    (target / "train.log").write_text("Val loss 2.750\n")

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._active_adapter_path = "/some/other/served"
    result = await f.parametric_self_rollback(adapter_id="pick99")
    assert result.status == ToolResultStatus.OK
    assert f._active_adapter_path == str(target)
    assert f._last_val_loss == pytest.approx(2.750)


async def test_rollback_unknown_adapter_id_fails(tmp_path):
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    (tmp_path / "parametric_self" / "candidates").mkdir(parents=True)
    result = await f.parametric_self_rollback(adapter_id="nope")
    assert result.status == ToolResultStatus.ERROR
    assert "No candidate adapter" in (result.error or "")


async def test_rollback_no_prior_fails(tmp_path):
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    (tmp_path / "parametric_self" / "candidates").mkdir(parents=True)
    result = await f.parametric_self_rollback()
    assert result.status == ToolResultStatus.ERROR
    assert "No prior promoted adapter" in (result.error or "")
