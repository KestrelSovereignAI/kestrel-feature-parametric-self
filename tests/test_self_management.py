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


async def test_nightly_training_refused_for_governed_agent():
    """Even with nightly training enabled (persisted/external), a governed agent
    must not self-modify via the sleep cycle (Incubator Principle, all paths)."""
    f = await _feature(_FakeStorage(), is_test_instance=True, storage_path="/x/kestrel_prime.db")
    f._training_enabled = True  # as if persisted config had enabled it
    result = await f.on_post_consolidation({"episodes_created": 5})
    assert result["trained"] is False
    assert result["promoted"] is False
    assert "Incubator Principle" in result["reason"]


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


@pytest.mark.parametrize("raw,expected", [
    ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ("true", True), ("1", True), (True, True), (False, False),
    # Defensive: a leaked `key=value` token from the positional parser.
    ("enabled=false", False), ("enabled=true", True),
])
async def test_set_enabled_coerces_string_booleans(raw, expected):
    """A command-path string like 'false' must disable, not enable (bool('false') is True)."""
    f = await _feature(_FakeStorage())
    f._training_enabled = not expected  # start from the opposite state
    result = await f.parametric_self_set_enabled(raw)
    assert result.status == ToolResultStatus.OK
    assert f._training_enabled is expected


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

    f._run_training_cycle_locked = _fake_cycle

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

    f._run_training_cycle_locked = _slow_cycle
    first = await f.parametric_self_train_now()
    assert first.status == ToolResultStatus.OK
    await asyncio.wait_for(started.wait(), timeout=2)
    busy = await f.parametric_self_train_now()
    assert busy.status == ToolResultStatus.ERROR
    assert "in progress" in (busy.error or "")
    release.set()
    await f._training_task


async def test_on_disable_cancels_in_flight_training_task():
    """A detached manual run must be cancelled on disable, not left to mutate state."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    f.agent.sleep_hooks = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_cycle(*, trigger):
        started.set()
        await release.wait()  # never released; cancellation must break this
        return {}

    f._run_training_cycle_locked = _slow_cycle
    # Track that the MLX subprocess(es) are also terminated on disable.
    cancel_all_called = asyncio.Event()

    async def _cancel_all():
        cancel_all_called.set()
        return 0

    f._adapter.cancel_all = _cancel_all

    result = await f.parametric_self_train_now()
    assert result.status == ToolResultStatus.OK
    await asyncio.wait_for(started.wait(), timeout=2)

    task = f._training_task
    await f.on_disable()
    assert f._training_task is None
    assert cancel_all_called.is_set()  # subprocess termination requested
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_on_disable_clears_guard_when_cancel_precedes_runner():
    """If disable cancels the task before _runner starts, the in-flight guard must
    still be cleared — otherwise a re-enabled instance refuses everything."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    f.agent.sleep_hooks = []

    async def _never_runs(*, trigger):
        return {}

    f._run_training_cycle_locked = _never_runs
    result = await f.parametric_self_train_now()
    assert result.status == ToolResultStatus.OK
    assert f._cycle_in_flight is True  # reserved synchronously by train_now
    # Disable BEFORE yielding to the loop, so _runner never starts.
    await f.on_disable()
    assert f._cycle_in_flight is False  # guard force-cleared on teardown
    assert f._training_task is None


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


@pytest.mark.parametrize("arg", ["pick99", "adapter_id=pick99"])
async def test_rollback_explicit_adapter_id(tmp_path, arg):
    work = tmp_path / "parametric_self"
    cands = work / "candidates"
    target = cands / "pick99"
    target.mkdir(parents=True)
    (target / "train.log").write_text("Val loss 2.750\n")

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._active_adapter_path = "/some/other/served"
    # Accept both the split form ("pick99") and a leaked key=value token.
    result = await f.parametric_self_rollback(adapter_id=arg)
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


@pytest.mark.parametrize("bad_id", ["../escape", "/etc", "a/b", "..", "."])
async def test_rollback_rejects_path_traversal(tmp_path, bad_id):
    """adapter_id must be a simple child name — never escape the candidates dir."""
    cands = tmp_path / "parametric_self" / "candidates"
    cands.mkdir(parents=True)
    # Create a sibling dir that a '../' could try to reach.
    (tmp_path / "parametric_self" / "escape").mkdir()
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    result = await f.parametric_self_rollback(adapter_id=bad_id)
    assert result.status == ToolResultStatus.ERROR
    assert f._active_adapter_path is None  # never repointed


async def test_rollback_refused_while_cycle_in_flight(tmp_path):
    """Rollback must not mutate served state while a training cycle is running."""
    cands = tmp_path / "parametric_self" / "candidates" / "abc123"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Val loss 2.500\n")
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._cycle_in_flight = True
    result = await f.parametric_self_rollback(adapter_id="abc123")
    assert result.status == ToolResultStatus.ERROR
    assert "in progress" in (result.error or "")
    assert f._active_adapter_path is None  # untouched


async def test_rollback_refuses_adapter_without_val_loss(tmp_path):
    """An incomplete candidate (no parseable val_loss) must not be served."""
    cands = tmp_path / "parametric_self" / "candidates" / "incomplete"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 10: training started...\n")  # no Val loss line
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    result = await f.parametric_self_rollback(adapter_id="incomplete")
    assert result.status == ToolResultStatus.ERROR
    assert "no parseable validation loss" in (result.error or "")
    assert f._active_adapter_path is None


async def test_nightly_cycle_skips_when_manual_run_in_flight():
    """on_post_consolidation must skip (not race) when a cycle is already running."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._training_enabled = True
    f._cycle_in_flight = True  # simulate a manual run holding the guard
    result = await f.on_post_consolidation({"episodes_created": 1})
    assert result["trained"] is False
    assert "in progress" in result["reason"]


# ----------------------------------------------------------------------
# Adoption / recovery path (candidate on disk, no served pointer)
# ----------------------------------------------------------------------

async def test_adapters_marks_recoverable_when_unserved(tmp_path):
    """A valid candidate with no served pointer is flagged recoverable."""
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 400: Val loss 2.688\n")
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    # No served adapter (the legacy/interrupted-run state).
    assert f._active_adapter_path is None

    result = await f.parametric_self_adapters()
    assert result.status == ToolResultStatus.OK
    by_id = {a["adapter_id"]: a for a in result.data["adapters"]}
    assert by_id["ded33cd017d9"]["recoverable"] is True
    assert by_id["ded33cd017d9"]["served"] is False
    assert result.data["recoverable_adapters"] == ["ded33cd017d9"]


async def test_status_exposes_recoverable_candidates(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 400: Val loss 2.688\n")
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))

    result = await f.parametric_self_status()
    assert result.status == ToolResultStatus.OK
    assert result.data["served_adapter"] is None
    assert result.data["recoverable_adapters"] == ["ded33cd017d9"]


async def test_adopt_persists_served_and_appends_adopt_history(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 400: Val loss 2.688\n")
    storage = _FakeStorage()
    f = await _feature(storage, storage_path=str(tmp_path / "kestrel_prime.db"))

    result = await f.parametric_self_adopt(adapter_id="ded33cd017d9")
    assert result.status == ToolResultStatus.OK
    assert f._active_adapter_path == str(cands)
    assert f._last_val_loss == pytest.approx(2.688)
    assert f._config_node_id() in storage.nodes  # served pointer persisted
    runs = await f._load_run_history()
    assert runs[-1]["trigger"] == "adopt"
    assert runs[-1]["adapter_path"] == str(cands)


@pytest.mark.parametrize("arg", ["ded33cd017d9", "adapter_id=ded33cd017d9"])
async def test_adopt_accepts_leaked_key_value_token(tmp_path, arg):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Val loss 2.688\n")
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    result = await f.parametric_self_adopt(adapter_id=arg)
    assert result.status == ToolResultStatus.OK
    assert f._active_adapter_path == str(cands)


async def test_adopt_refuses_adapter_without_val_loss(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "incomplete"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 10: training started...\n")  # no Val loss
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    result = await f.parametric_self_adopt(adapter_id="incomplete")
    assert result.status == ToolResultStatus.ERROR
    assert "no parseable validation loss" in (result.error or "")
    assert f._active_adapter_path is None


async def test_adopt_unknown_adapter_id_fails(tmp_path):
    (tmp_path / "parametric_self" / "candidates").mkdir(parents=True)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    result = await f.parametric_self_adopt(adapter_id="nope")
    assert result.status == ToolResultStatus.ERROR
    assert "No candidate adapter" in (result.error or "")
    assert f._active_adapter_path is None


@pytest.mark.parametrize("bad_id", ["../escape", "/etc", "a/b", "..", "."])
async def test_adopt_rejects_path_traversal(tmp_path, bad_id):
    cands = tmp_path / "parametric_self" / "candidates"
    cands.mkdir(parents=True)
    (tmp_path / "parametric_self" / "escape").mkdir()
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    result = await f.parametric_self_adopt(adapter_id=bad_id)
    assert result.status == ToolResultStatus.ERROR
    assert f._active_adapter_path is None


async def test_adopt_refused_for_governed_agent(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Val loss 2.688\n")
    f = await _feature(
        _FakeStorage(), is_test_instance=True, storage_path=str(tmp_path / "kestrel_prime.db"),
    )
    result = await f.parametric_self_adopt(adapter_id="ded33cd017d9")
    assert result.status == ToolResultStatus.ERROR
    assert "Incubator Principle" in (result.error or "")
    assert f._active_adapter_path is None


async def test_adopt_rejects_candidate_failing_fidelity_gate(tmp_path):
    """A candidate whose val_loss exceeds the ceiling must not be adopted."""
    cands = tmp_path / "parametric_self" / "candidates" / "toobad"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Val loss 3.500\n")  # > max_val_loss (3.0)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    result = await f.parametric_self_adopt(adapter_id="toobad")
    assert result.status == ToolResultStatus.ERROR
    assert "fidelity gate" in (result.error or "")
    assert f._active_adapter_path is None


async def test_adopt_refused_while_cycle_in_flight(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Val loss 2.688\n")
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._cycle_in_flight = True
    result = await f.parametric_self_adopt(adapter_id="ded33cd017d9")
    assert result.status == ToolResultStatus.ERROR
    assert "in progress" in (result.error or "")
    assert f._active_adapter_path is None
