"""Agent introspection + sovereign-class self-management tools (epic #10).

Covers the two view tools (history, adapters), the three mutation tools
(train_now, set_enabled, rollback), and — critically — the Incubator-Principle
gate: governed/test instances must be refused, sovereign-class agents allowed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_feature_parametric_self import ParametricSelfFeature


def _snapshot():
    assertion = SimpleNamespace(
        assertion_id="assertion:test", revision_id="revision:test",
        subject=SimpleNamespace(value="https://example.test/agent"),
        predicate=SimpleNamespace(value="https://example.test/lesson"),
        object=SimpleNamespace(lexical_form="lesson"),
    )
    return SimpleNamespace(
        verified=True, examples=(SimpleNamespace(
            assertion=assertion, content_hash="sha256:test", source_occurrences=(),
            decision=SimpleNamespace(included=True, reason=SimpleNamespace(value="included")),
        ),), snapshot_hash="sha256:snapshot", policy=SimpleNamespace(digest="sha256:policy"),
        tenant_id="tenant:test",
        checkpoint=SimpleNamespace(tenant_id="tenant:test", generation=1, latest_event_id="event:1"),
        capability_versions={"semantic_maintenance": "1"},
    )


def _write_governed_manifest(candidate):
    raw = {
        "schema_version": 1, "corpus_policy_version": "parametric-self-corpus-v1",
        "policy_digest": "sha256:policy", "snapshot_hash": "sha256:snapshot",
        "semantic_checkpoint": {"tenant_id": "tenant:test", "generation": 1, "event_id": "event:1"},
        "capability_versions": {"semantic_maintenance": "1"},
        "counts": {"total": 1, "train": 1, "valid": 0, "reflection": 0, "governed_assertion": 1},
        "examples": [{"example_id": "example:1", "source": "governed_assertion", "split": "train", "lineage": {
            "assertion_id": "assertion:test", "revision_id": "revision:test", "content_hash": "sha256:test",
            "source_occurrence_ids": [], "eligibility": "included",
        }}],
    }
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    raw["manifest_hash"] = "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
    (candidate / "corpus_manifest.json").write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")))


class _FakeStorage:
    """In-memory stand-in for the agent graph store (add_node/get_node)."""

    def __init__(self):
        self.nodes = {}

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def governed_assertion_corpus_snapshot(self, **_kwargs):
        return _snapshot()

    async def governed_assertion_corpus_changes_since(self, _snapshot_value, **_kwargs):
        return SimpleNamespace(tombstones=())


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
    f._governed_corpus_policy = SimpleNamespace(digest="sha256:policy")
    return f


def _stamp_adapter_receipt(feature, candidate) -> None:
    manifest, error = feature._manifest_lineage(str(candidate))
    assert error is None
    receipt = feature._manifest_receipt_stamp(manifest or {})
    assert receipt is not None
    feature._adapter_lineage[str(candidate)] = {**receipt, "state": "candidate"}


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
    assert result.data["active_run"]["state"] == "in_progress"
    assert result.data["active_run"]["trigger"] == "manual"
    assert result.data["active_run"] == f._active_run
    runs = await f._load_run_history()
    assert runs[-1]["run_id"] == result.data["active_run"]["run_id"]
    assert runs[-1]["state"] == "in_progress"
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
        return SimpleNamespace(all_stopped=True)

    f._adapter.cancel_all = _cancel_all

    result = await f.parametric_self_train_now()
    assert result.status == ToolResultStatus.OK
    await asyncio.wait_for(started.wait(), timeout=2)

    task = f._training_task
    await f.on_disable()
    assert f._training_task is None
    assert cancel_all_called.is_set()  # subprocess termination requested
    assert f._training_shutdown_incomplete is None
    assert f._cycle_in_flight is False
    assert (await f.parametric_self_status()).data["training_shutdown_incomplete"] is None
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
    assert f._active_run is not None
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "in_progress"
    # Disable BEFORE yielding to the loop, so _runner never starts.
    await f.on_disable()
    assert f._cycle_in_flight is False  # guard force-cleared on teardown
    assert f._training_task is None
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "interrupted"


async def test_on_disable_surfaces_unconfirmed_trainer_shutdown():
    """Disable must not claim a live child stopped when bulk confirmation fails."""
    from kestrel_feature_parametric_self.cycle import TrainingShutdownIncomplete

    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    f.agent.sleep_hooks = []
    started = asyncio.Event()

    async def _unconfirmed_cycle(*, trigger):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise TrainingShutdownIncomplete("trainer stop could not be confirmed; corpus retained")

    async def _unconfirmed_cancel_all():
        return SimpleNamespace(all_stopped=False)

    f._run_training_cycle_locked = _unconfirmed_cycle
    f._adapter.cancel_all = _unconfirmed_cancel_all
    result = await f.parametric_self_train_now()
    assert result.status == ToolResultStatus.OK
    task = f._training_task
    await asyncio.wait_for(started.wait(), timeout=2)
    await f.on_disable()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert f._training_task is None
    assert f._cycle_in_flight is True
    assert f._active_run is not None
    assert f._active_run["state"] == "shutdown_incomplete"
    status = await f.parametric_self_status()
    assert "shutdown incomplete" in status.confirmation
    assert "shutdown incomplete" in status.data["training_shutdown_incomplete"]
    assert status.data["active_run"]["state"] == "shutdown_incomplete"
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "shutdown_incomplete"


async def test_confirmed_bulk_shutdown_resolves_prior_incomplete_run():
    """A later positive bulk confirmation clears the block and terminalizes history."""
    from kestrel_feature_parametric_self.cycle import TrainingShutdownIncomplete

    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    started = asyncio.Event()

    async def _unconfirmed_cycle(*, trigger):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise TrainingShutdownIncomplete("trainer stop could not be confirmed; corpus retained")

    async def _confirmed_cancel_all():
        return SimpleNamespace(all_stopped=True)

    f._run_training_cycle_locked = _unconfirmed_cycle
    f._adapter.cancel_all = _confirmed_cancel_all
    started_result = await f.parametric_self_train_now()
    assert started_result.status == ToolResultStatus.OK
    task = f._training_task
    await asyncio.wait_for(started.wait(), timeout=2)
    await f.on_disable()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert f._training_shutdown_incomplete is None
    assert f._cycle_in_flight is False
    assert f._active_run is None
    status = await f.parametric_self_status()
    assert status.data["training_shutdown_incomplete"] is None
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "interrupted"
    assert "bulk trainer stop confirmed" in runs[-1]["reason"]


async def test_nightly_unconfirmed_shutdown_blocks_like_manual(tmp_path):
    """Nightly cancellation enters the same durable safety state as train_now."""
    from kestrel_feature_parametric_self.cycle import TrainingShutdownIncomplete

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._adapter.is_available = lambda: True

    async def _unconfirmed_cycle(*, trigger):
        active = await f._begin_active_run(trigger=trigger, work_dir=str(tmp_path / "work"))
        assert active["trigger"] == "nightly"
        raise TrainingShutdownIncomplete("trainer stop could not be confirmed; corpus retained")

    f._run_training_cycle_locked = _unconfirmed_cycle
    with pytest.raises(TrainingShutdownIncomplete):
        await f._run_training_cycle(trigger="nightly")

    assert f._cycle_in_flight is True
    assert f._active_run is not None
    assert f._active_run["state"] == "shutdown_incomplete"
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "shutdown_incomplete"
    blocked = await f.parametric_self_train_now()
    assert blocked.status == ToolResultStatus.ERROR
    assert "shutdown incomplete" in (blocked.error or "")

    # The durable marker survives feature reconstruction and remains a block.
    restarted = ParametricSelfFeature(agent=f.agent)
    await restarted.initialize()
    restarted.agent.get_feature = MagicMock(return_value=restarted)
    await restarted.post_all_features_loaded(restarted.agent)
    assert restarted._training_shutdown_incomplete is not None
    assert restarted._cycle_in_flight is True
    assert (await restarted.parametric_self_status()).data["training_shutdown_incomplete"]


async def test_train_now_cancellation_during_history_reservation_recovers():
    """Cancelling the command while its run record is awaited releases all state."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    entered = asyncio.Event()
    never = asyncio.Event()
    original_append = f._append_run_history

    async def _blocked_append(entry):
        entered.set()
        await never.wait()

    f._append_run_history = _blocked_append
    command = asyncio.create_task(f.parametric_self_train_now())
    await asyncio.wait_for(entered.wait(), timeout=2)
    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command

    assert f._cycle_in_flight is False
    assert f._training_task is None
    assert f._active_run is None
    progress = await f.parametric_self_progress()
    assert progress.data == {"active_run": None}
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "interrupted"

    f._append_run_history = original_append

    async def _fast_cycle(*, trigger):
        return {"trained": False, "promoted": False, "reason": "test skip"}

    f._run_training_cycle_locked = _fast_cycle
    recovered = await f.parametric_self_train_now()
    assert recovered.status == ToolResultStatus.OK
    await f._training_task
    runs = await f._load_run_history()
    assert [run["state"] for run in runs] == ["interrupted", "skipped"]


async def test_cancelled_detached_manual_run_is_terminal_and_recoverable():
    """External task cancellation cannot strand progress/history as in-progress."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    started = asyncio.Event()

    async def _slow_cycle(*, trigger):
        started.set()
        await asyncio.Event().wait()

    f._run_training_cycle_locked = _slow_cycle
    result = await f.parametric_self_train_now()
    assert result.status == ToolResultStatus.OK
    task = f._training_task
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert f._cycle_in_flight is False
    assert f._training_task is None
    assert f._active_run is None
    progress = await f.parametric_self_progress()
    assert progress.data == {"active_run": None}
    runs = await f._load_run_history()
    assert len(runs) == 1
    assert runs[0]["state"] == "interrupted"
    assert runs[0]["reason"] == "run cancelled"

    async def _fast_cycle(*, trigger):
        return {"trained": False, "promoted": False, "reason": "test skip"}

    f._run_training_cycle_locked = _fast_cycle
    recovered = await f.parametric_self_train_now()
    assert recovered.status == ToolResultStatus.OK
    await f._training_task
    runs = await f._load_run_history()
    assert [run["state"] for run in runs] == ["interrupted", "skipped"]


async def test_disable_during_manual_record_creation_prevents_runner_launch():
    """Disable invalidates a suspended train_now command before it can detach."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    entered = asyncio.Event()
    release = asyncio.Event()
    cycle_calls = []

    async def _blocked_append(entry):
        entered.set()
        await release.wait()

    async def _should_not_run(*, trigger):
        cycle_calls.append(trigger)
        return {"trained": True, "promoted": False}

    f._append_run_history = _blocked_append
    f._run_training_cycle_locked = _should_not_run
    command = asyncio.create_task(f.parametric_self_train_now())
    await asyncio.wait_for(entered.wait(), timeout=2)
    disable = asyncio.create_task(f.on_disable())
    await asyncio.sleep(0)
    release.set()

    result = await command
    await disable
    assert result.status == ToolResultStatus.ERROR
    assert "did not start" in (result.error or "")
    assert cycle_calls == []
    assert f._training_task is None
    assert f._cycle_in_flight is False
    assert f._active_run is None
    progress = await f.parametric_self_progress()
    assert progress.data == {"active_run": None}
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "interrupted"


async def test_rollback_default_to_previous_promoted(tmp_path):
    work = tmp_path / "parametric_self"
    cands = work / "candidates"
    old = cands / "old111"
    new = cands / "new222"
    for d, loss in ((old, "2.900"), (new, "2.600")):
        d.mkdir(parents=True)
        (d / "train.log").write_text(f"Val loss {loss}\n")
        _write_governed_manifest(d)

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, old)
    _stamp_adapter_receipt(f, new)
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
    _write_governed_manifest(target)

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, target)
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

async def test_adapters_leaves_untracked_candidates_inspection_only(tmp_path):
    """A valid-looking legacy candidate is visible but cannot be served."""
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 400: Val loss 2.688\n")
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    # No served adapter (the legacy/interrupted-run state).
    assert f._active_adapter_path is None

    result = await f.parametric_self_adapters()
    assert result.status == ToolResultStatus.OK
    by_id = {a["adapter_id"]: a for a in result.data["adapters"]}
    assert by_id["ded33cd017d9"]["recoverable"] is False
    assert by_id["ded33cd017d9"]["served"] is False
    assert "durable adapter lineage receipt unavailable" in by_id["ded33cd017d9"]["quarantined_reason"]
    assert result.data["recoverable_adapters"] == []


async def test_status_keeps_untracked_candidates_non_serving(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 400: Val loss 2.688\n")
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))

    result = await f.parametric_self_status()
    assert result.status == ToolResultStatus.OK
    assert result.data["served_adapter"] is None
    assert result.data["recoverable_adapters"] == []


async def test_adopt_persists_served_and_appends_adopt_history(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 400: Val loss 2.688\n")
    _write_governed_manifest(cands)
    storage = _FakeStorage()
    f = await _feature(storage, storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)

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
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
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
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
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


# ----------------------------------------------------------------------
# In-progress run surface (issue #17)
# ----------------------------------------------------------------------

async def test_progress_none_when_idle():
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    result = await f.parametric_self_progress()
    assert result.status == ToolResultStatus.OK
    assert result.data["active_run"] is None
    assert "No parametric-self training run" in result.confirmation


async def test_progress_reports_active_run_with_live_iter(tmp_path):
    """An active run surfaces its run_id and the latest log iter, marking the
    intermediate val_loss as non-terminal (the core of issue #17)."""
    work = tmp_path / "parametric_self"
    cands = work / "candidates" / "abc123def456"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text(
        "Iter 60: Val loss 8.977\nIter 230: Val loss 1.86\nIter 400: Val loss 2.69\n"
    )
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._active_run = {
        "run_id": "run0001",
        "adapter_id": "abc123def456",
        "trigger": "manual",
        "started_at": "2026-06-18T01:00:00+00:00",
        "state": "in_progress",
        "adapter_path": str(cands),
    }

    result = await f.parametric_self_progress()
    assert result.status == ToolResultStatus.OK
    run = result.data["active_run"]
    assert run["run_id"] == "run0001"
    assert run["state"] == "in_progress"
    assert run["last_seen_iter"] == 400
    assert run["latest_val_loss"] == pytest.approx(2.69)


async def test_status_exposes_active_run(tmp_path):
    work = tmp_path / "parametric_self"
    cands = work / "candidates" / "abc123def456"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 400: Val loss 2.69\n")
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._active_run = {
        "run_id": "run0001", "adapter_id": "abc123def456", "trigger": "manual",
        "started_at": "2026-06-18T01:00:00+00:00", "state": "in_progress",
        "adapter_path": str(cands),
    }
    result = await f.parametric_self_status()
    assert result.status == ToolResultStatus.OK
    assert result.data["active_run"]["run_id"] == "run0001"
    assert "in progress" in result.confirmation


async def test_active_candidate_marked_in_progress_not_recoverable(tmp_path):
    """A candidate of the active run must read in_progress, never recoverable,
    so an intermediate snapshot isn't presented as adoptable."""
    work = tmp_path / "parametric_self"
    cands = work / "candidates" / "abc123def456"
    cands.mkdir(parents=True)
    (cands / "train.log").write_text("Iter 230: Val loss 1.86\n")  # intermediate
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    assert f._active_adapter_path is None  # nothing served yet
    f._active_run = {
        "run_id": "run0001", "adapter_id": "abc123def456", "trigger": "manual",
        "started_at": "2026-06-18T01:00:00+00:00", "state": "in_progress",
        "adapter_path": str(cands),
    }
    result = await f.parametric_self_adapters()
    by_id = {a["adapter_id"]: a for a in result.data["adapters"]}
    assert by_id["abc123def456"]["in_progress"] is True
    assert by_id["abc123def456"]["recoverable"] is False
    assert result.data["recoverable_adapters"] == []


async def test_cycle_records_in_progress_then_completed(tmp_path):
    """The cycle appends an in_progress entry up front and updates it in place
    on completion — one durable record, not a completion-only append."""
    db = _db_path_with_corpus(tmp_path)
    f = await _feature(_FakeStorage(), storage_path=db)
    f._adapter.is_available = lambda: True
    seen_states = []

    real_run = f._run_training_cycle_locked

    # Capture history state mid-run by patching run_nightly_cycle via the module.
    import kestrel_feature_parametric_self.feature as feat_mod

    async def _fake_run_cycle(**kwargs):
        runs = await f._load_run_history()
        seen_states.append(runs[-1]["state"])  # in_progress while running
        from kestrel_feature_parametric_self.cycle import CycleResult
        from kestrel_feature_parametric_self.corpus import build_corpus
        candidate = kwargs["work_dir"] + "/candidates/" + kwargs["adapter_id"]
        stats = build_corpus(
            kwargs["db_path"], kwargs["work_dir"] + "/corpus",
            governed_snapshot=kwargs["governed_snapshot"], manifest_dir=candidate,
        )
        return CycleResult(
            trained=True, promoted=True, reason="ok", val_loss=1.2,
            promoted_adapter_path=candidate,
            corpus_train=5, corpus_manifest_path=stats.manifest_path,
            corpus_manifest_hash=stats.manifest_hash,
            semantic_checkpoint_generation=stats.semantic_checkpoint_generation,
            semantic_checkpoint_id=stats.semantic_checkpoint_id,
            corpus_snapshot_hash=stats.snapshot_hash,
            corpus_policy_digest=stats.policy_digest,
            assertion_lineage=stats.assertion_lineage,
        )

    orig = feat_mod.run_nightly_cycle
    feat_mod.run_nightly_cycle = _fake_run_cycle
    async def _valid_lineage(*_args, **_kwargs):
        return None
    f._verify_adapter_lineage = _valid_lineage
    try:
        outcome = await real_run(trigger="manual")
    finally:
        feat_mod.run_nightly_cycle = orig

    assert outcome["promoted"] is True
    assert seen_states == ["in_progress"]
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "completed"
    assert runs[-1]["trained"] is True
    assert f._active_run is None


async def test_cycle_marks_failed_on_exception(tmp_path):
    db = _db_path_with_corpus(tmp_path)
    f = await _feature(_FakeStorage(), storage_path=db)
    # Reach the patched training exception path on Linux CI as well as macOS.
    f._adapter.is_available = lambda: True
    import kestrel_feature_parametric_self.feature as feat_mod

    async def _boom(**kwargs):
        raise RuntimeError("kaboom")

    orig = feat_mod.run_nightly_cycle
    feat_mod.run_nightly_cycle = _boom
    try:
        with pytest.raises(RuntimeError):
            await f._run_training_cycle_locked(trigger="manual")
    finally:
        feat_mod.run_nightly_cycle = orig

    runs = await f._load_run_history()
    assert runs[-1]["state"] == "failed"
    assert "kaboom" in runs[-1]["reason"]
    assert f._active_run is None


async def test_reconcile_marks_stale_in_progress_interrupted():
    storage = _FakeStorage()
    f = await _feature(storage)
    await f._append_run_history({
        "run_id": "stale1", "trigger": "manual", "state": "in_progress",
        "trained": False, "promoted": False,
    })
    await f._reconcile_stale_runs()
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "interrupted"


def _db_path_with_corpus(tmp_path) -> str:
    """Minimal storage_path; run_nightly_cycle is faked so the DB is unused."""
    return str(tmp_path / "kestrel_prime.db")
