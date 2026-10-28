"""Agent introspection + sovereign-class self-management tools (epic #10).

Covers the two view tools (history, adapters), the three mutation tools
(train_now, set_enabled, rollback), and — critically — the Incubator-Principle
gate: governed/test instances must be refused, sovereign-class agents allowed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_feature_parametric_self import (
    Baseline,
    Checkpoint,
    FidelityGate,
    ParametricSelfFeature,
    TextLoRAConfig,
    ValSet,
)


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


def _write_governed_manifest(candidate, assertion_id="assertion:test"):
    raw = {
        "schema_version": 1, "corpus_policy_version": "parametric-self-corpus-v1",
        "policy_digest": "sha256:policy", "snapshot_hash": "sha256:snapshot",
        "semantic_checkpoint": {"tenant_id": "tenant:test", "generation": 1, "event_id": "event:1"},
        "capability_versions": {"semantic_maintenance": "1"},
        "counts": {"total": 1, "train": 1, "valid": 0, "reflection": 0, "governed_assertion": 1},
        "examples": [{"example_id": "example:1", "source": "governed_assertion", "split": "train", "lineage": {
            "assertion_id": assertion_id, "revision_id": "revision:test", "content_hash": "sha256:test",
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


ITER_200 = "0000200_adapters.safetensors"
ITER_400 = "0000400_adapters.safetensors"
WEIGHT_FILES = ("0000100_adapters.safetensors", ITER_200, "0000300_adapters.safetensors", ITER_400, "adapters.safetensors")
JUNE_LOG = "Iter 1: Val loss 8.977, x\nIter 200: Val loss 1.866, x\nIter 400: Val loss 2.688, x\n"
# The same curve as mlx-lm 0.32's training UI prints it (cli_ui.TrainUI): no "Val loss" line.
JUNE_LOG_MLX_032 = (
    "     1    val 8.977    41.20s\n"
    "   200    val 1.866    40.10s\n"
    "  save  0000200_adapters.safetensors\n"
    "   400    val 2.688    40.00s\n"
    "  save  0000400_adapters.safetensors\n"
)
IMPROVING_LOG = "Iter 1: Val loss 8.977, x\nIter 200: Val loss 1.866, x\nIter 400: Val loss 1.800, x\n"
# What the candidate's training run scored each checkpoint file at.
JUNE_SCORES = {ITER_200: 1.866, ITER_400: 2.688}
IMPROVING_SCORES = {ITER_200: 1.866, ITER_400: 1.800}


def _serve_governed(feature, served, assertion_id="assertion:test") -> None:
    """Serve ``served`` at iter 200 with the governed lineage receipt it was trained under."""
    _write_run(served, IMPROVING_LOG)
    _write_governed_manifest(served, assertion_id)
    _stamp_adapter_receipt(feature, served)
    feature._active_adapter_path, feature._active_checkpoint_id = str(served), ITER_200


def _write_run(candidate, log, files=WEIGHT_FILES):
    """A finished candidate dir: its train.log and the weights mlx_lm saved."""
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "train.log").write_text(log)
    for name in files:
        (candidate / name).write_bytes(b"weights")


async def _record_training_gate(
    feature, candidate, scores=IMPROVING_SCORES, *, baseline=("base", 8.977),
    digest="sha256:" + "cd" * 32, examples=7, base_model=TextLoRAConfig().base_model,
):
    """The gate record the candidate's training run left: every score adopt re-decides from."""
    checkpoints = tuple(Checkpoint(name, int(name[:7]), loss) for name, loss in scores.items())
    record = FidelityGate().evaluate(
        checkpoints=checkpoints, baseline=Baseline(*baseline), val_set=ValSet(digest, examples, 4),
        base_model=base_model,
    ).to_record()
    await feature._append_run_history({
        "run_id": f"train-{candidate.name}", "trigger": "nightly", "state": "completed",
        "trained": True, "promoted": False, "adapter_path": str(candidate), "gate": record,
    })


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
    for name in ("aaa111", "bbb222"):
        _write_run(cands / name, "Iter 400: Val loss 3.100\n")
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._active_adapter_path, f._active_checkpoint_id, f._last_val_loss = str(cands / "aaa111"), ITER_200, 2.601
    await _record_training_gate(f, cands / "bbb222", JUNE_SCORES)

    result = await f.parametric_self_adapters()
    assert result.status == ToolResultStatus.OK
    by_id = {a["adapter_id"]: a for a in result.data["adapters"]}
    assert by_id["aaa111"]["served"] is True
    assert by_id["bbb222"]["served"] is False
    # The served adapter shows its served checkpoint and score; every other
    # candidate the scores its training run's gate recorded. Never the log.
    assert (by_id["aaa111"]["checkpoint_id"], by_id["aaa111"]["val_loss"]) == (ITER_200, 2.601)
    assert (by_id["bbb222"]["checkpoint_id"], by_id["bbb222"]["val_loss"]) == (ITER_200, 1.866)
    assert (by_id["bbb222"]["final_checkpoint_id"], by_id["bbb222"]["final_val_loss"]) == (ITER_400, 2.688)


# ----------------------------------------------------------------------
# Incubator-Principle gate
# ----------------------------------------------------------------------

async def test_mutation_tools_refused_for_governed_agent():
    f = await _feature(_FakeStorage(), is_test_instance=True)
    for result in (
        await f.parametric_self_train_now(),
        await f.parametric_self_set_enabled(True),
        await f.parametric_self_recover_shutdown(
            confirmed_process_absent=True,
            evidence="checked process table; no trainer remains",
        ),
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


async def test_confirmed_bulk_shutdown_resolves_prior_incomplete_run_and_deletes_corpus(tmp_path):
    """Same-process stop proof terminalizes history and releases retained input."""
    from kestrel_feature_parametric_self.cycle import TrainingShutdownIncomplete

    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._work_dir = str(tmp_path / "parametric_self")
    f._adapter.is_available = lambda: True
    started = asyncio.Event()
    retained = tmp_path / "parametric_self" / "corpus" / "run-1"
    retained.mkdir(parents=True)
    (retained / "train.jsonl").write_text('{"text":"private"}\n')
    (retained / "valid.jsonl").write_text('{"text":"private"}\n')

    async def _unconfirmed_cycle(*, trigger):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise TrainingShutdownIncomplete(
                "trainer stop could not be confirmed; corpus retained",
                corpus_path=str(retained),
            )

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
    assert not (retained / "train.jsonl").exists()
    assert not (retained / "valid.jsonl").exists()


async def test_shutdown_without_bulk_confirmation_keeps_prior_incomplete_block_and_corpus(tmp_path):
    """A missing bulk cancellation capability is unknown, never clean shutdown."""
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    retained = tmp_path / "parametric_self" / "corpus" / "unresolved"
    retained.mkdir(parents=True)
    (retained / "train.jsonl").write_text('{"text":"private"}\n')
    f._active_run = {
        "run_id": "unresolved", "adapter_id": "adapter", "trigger": "manual",
        "state": "shutdown_incomplete", "adapter_path": "/tmp/adapter",
        "retained_corpus_path": str(retained),
    }
    await f._append_run_history({
        "run_id": "unresolved", "trigger": "manual", "state": "shutdown_incomplete",
        "reason": "training shutdown incomplete: prior stop was unconfirmed",
        "retained_corpus_path": str(retained),
    })
    f._training_shutdown_incomplete = "training shutdown incomplete: prior stop was unconfirmed"
    f._adapter.cancel_all = None

    await f.on_disable()
    assert f._training_shutdown_incomplete is not None
    assert f._cycle_in_flight is True
    assert f._active_run is not None
    assert (await f._load_run_history())[-1]["state"] == "shutdown_incomplete"
    assert (retained / "train.jsonl").exists()


async def test_same_instance_reenable_keeps_process_provenance_for_recovery():
    """Reloading a live feature must not turn its own stop proof into unknown."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f.agent.get_feature = MagicMock(return_value=f)
    f._active_run = {
        "run_id": "same-instance", "adapter_id": "adapter", "trigger": "manual",
        "state": "shutdown_incomplete", "adapter_path": "/tmp/adapter",
    }
    await f._append_run_history({
        "run_id": "same-instance", "trigger": "manual", "state": "shutdown_incomplete",
        "reason": "training shutdown incomplete: prior stop was unconfirmed",
    })
    f._training_shutdown_incomplete = "training shutdown incomplete: prior stop was unconfirmed"

    async def _unconfirmed_cancel_all():
        return SimpleNamespace(all_stopped=False)

    async def _confirmed_cancel_all():
        return SimpleNamespace(all_stopped=True)

    f._adapter.cancel_all = _unconfirmed_cancel_all
    await f.on_disable()
    assert f._training_shutdown_incomplete is not None

    await f.post_all_features_loaded(f.agent)
    assert f._shutdown_recovery_requires_external_confirmation is False
    f._adapter.cancel_all = _confirmed_cancel_all
    await f.on_disable()
    assert f._training_shutdown_incomplete is None
    assert f._cycle_in_flight is False
    assert (await f._load_run_history())[-1]["state"] == "interrupted"


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

    async def _confirmed_cancel_all():
        return SimpleNamespace(all_stopped=True)

    restarted._adapter.cancel_all = _confirmed_cancel_all
    await restarted.on_disable()
    # A fresh adapter's all_stopped result may only describe its own empty job
    # set; it is not proof that the prior process's child exited.
    assert restarted._training_shutdown_incomplete is not None
    assert restarted._shutdown_recovery_requires_external_confirmation is True
    assert restarted._cycle_in_flight is True
    unresolved_runs = await restarted._load_run_history()
    assert unresolved_runs[-1]["state"] == "shutdown_incomplete"

    final = ParametricSelfFeature(agent=f.agent)
    await final.initialize()
    final.agent.get_feature = MagicMock(return_value=final)
    await final.post_all_features_loaded(final.agent)
    assert final._training_shutdown_incomplete is not None
    assert final._cycle_in_flight is True


async def test_restored_shutdown_requires_explicit_sovereign_recovery_evidence(tmp_path):
    """A restart block has one explicit, auditable recovery path—not a silent ack."""
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    retained = tmp_path / "parametric_self" / "corpus" / "old-run"
    retained.mkdir(parents=True)
    (retained / "train.jsonl").write_text('{"text":"private"}\n')
    await f._append_run_history({
        "run_id": "old-run", "trigger": "nightly", "state": "shutdown_incomplete",
        "reason": "training shutdown incomplete: old process was not confirmed stopped",
        "retained_corpus_path": str(retained),
    })

    restarted = ParametricSelfFeature(agent=f.agent)
    await restarted.initialize()
    restarted.agent.get_feature = MagicMock(return_value=restarted)
    await restarted.post_all_features_loaded(restarted.agent)
    status = await restarted.parametric_self_status()
    assert status.data["shutdown_recovery_requires_external_confirmation"] is True
    assert "!parametric-self-recover-shutdown" in status.confirmation

    refused = await restarted.parametric_self_recover_shutdown()
    assert refused.status == ToolResultStatus.ERROR
    assert "confirmed_process_absent=true" in (refused.error or "")
    assert restarted._training_shutdown_incomplete is not None
    assert (retained / "train.jsonl").exists()

    recovered = await restarted.parametric_self_recover_shutdown(
        confirmed_process_absent=True,
        evidence="ps check at 2026-07-31 confirmed no prior mlx_lm.lora process",
    )
    assert recovered.status == ToolResultStatus.OK
    assert restarted._training_shutdown_incomplete is None
    assert restarted._shutdown_recovery_requires_external_confirmation is False
    assert restarted._cycle_in_flight is False
    assert not (retained / "train.jsonl").exists()
    runs = await restarted._load_run_history()
    assert runs[-1]["state"] == "interrupted"
    assert "sovereign operator verified" in runs[-1]["reason"]

    final = ParametricSelfFeature(agent=f.agent)
    await final.initialize()
    final.agent.get_feature = MagicMock(return_value=final)
    await final.post_all_features_loaded(final.agent)
    assert final._training_shutdown_incomplete is None
    assert final._shutdown_recovery_requires_external_confirmation is False
    assert final._cycle_in_flight is False


async def test_disable_fences_sleep_cycle_waiting_to_launch():
    """A sleep dispatch already queued on the lifecycle lock cannot start post-disable."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    launches = []

    async def _should_not_launch(*, trigger):
        launches.append(trigger)
        return {"trained": True, "promoted": False}

    f._run_training_cycle_locked = _should_not_launch
    await f._manual_run_lock.acquire()
    nightly = asyncio.create_task(f._run_training_cycle(trigger="nightly"))
    await asyncio.sleep(0)
    disable = asyncio.create_task(f.on_disable())
    await asyncio.sleep(0)
    f._manual_run_lock.release()

    outcome = await nightly
    await disable
    assert outcome["trained"] is False
    assert outcome["reason"] == "parametric-self feature is disabled"
    assert launches == []
    assert f._cycle_in_flight is False


async def test_poll_timeout_keeps_manual_lifecycle_blocked_until_confirmed_stop(tmp_path):
    """A returned poll timeout is nonterminal while the trainer may still write."""
    from kestrel_feature_parametric_self.cycle import TrainingStillActive

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    f._adapter.is_available = lambda: True
    retained = tmp_path / "parametric_self" / "corpus" / "still-live"
    retained.mkdir(parents=True)
    (retained / "train.jsonl").write_text('{"text":"private"}\n')

    async def _timed_out_cycle(*, trigger):
        raise TrainingStillActive(
            "training still active after poll timeout; corpus retained",
            corpus_path=str(retained),
        )

    f._run_training_cycle_locked = _timed_out_cycle
    started = await f.parametric_self_train_now()
    assert started.status == ToolResultStatus.OK
    await f._training_task
    assert f._cycle_in_flight is True
    assert f._active_run is not None
    assert f._active_run["state"] == "shutdown_incomplete"
    assert "still active" in f._training_shutdown_incomplete
    blocked = await f.parametric_self_train_now()
    assert blocked.status == ToolResultStatus.ERROR
    assert "still active" in (blocked.error or "")

    # No affirmative bulk capability means disable must preserve the block.
    f._adapter.cancel_all = None
    await f.on_disable()
    assert f._cycle_in_flight is True
    assert (await f._load_run_history())[-1]["state"] == "shutdown_incomplete"
    assert (retained / "train.jsonl").exists()

    # This is a same-process stop failure, so an operator-looking string cannot
    # bypass the adapter-specific stop proof or remove corpus a child may read.
    refused = await f.parametric_self_recover_shutdown(
        confirmed_process_absent=True,
        evidence="ps check at 2026-07-31 showed no prior trainer process",
    )
    assert refused.status == ToolResultStatus.ERROR
    assert "only for an incomplete shutdown restored" in (refused.error or "")
    assert f._training_shutdown_incomplete is not None
    assert f._cycle_in_flight is True
    assert (await f._load_run_history())[-1]["state"] == "shutdown_incomplete"
    assert (retained / "train.jsonl").exists()


async def test_disable_cancels_owned_nightly_task_not_sleep_owner():
    """The sleep dispatcher survives while its feature-owned nightly child stops."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._training_enabled = True
    f._adapter.is_available = lambda: True
    started = asyncio.Event()

    async def _slow_cycle(*, trigger):
        await f._begin_active_run(trigger=trigger, work_dir="/tmp/parametric-self-test")
        started.set()
        await asyncio.Event().wait()

    async def _confirmed_cancel_all():
        return SimpleNamespace(all_stopped=True)

    f._run_training_cycle_locked = _slow_cycle
    f._adapter.cancel_all = _confirmed_cancel_all
    sleep_owner = asyncio.create_task(f.on_post_consolidation({"episodes_created": 1}))
    await asyncio.wait_for(started.wait(), timeout=2)
    child = f._cycle_task
    assert child is not None and child is not sleep_owner
    await f.on_disable()
    outcome = await sleep_owner
    assert "interrupted" in outcome["reason"]
    assert not sleep_owner.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await child
    assert (await f._load_run_history())[-1]["state"] == "interrupted"


async def test_external_sleep_cancellation_propagates_and_terminalizes_nightly_run():
    """Core cancellation is not a feature disable and must remain observable."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._training_enabled = True
    f._adapter.is_available = lambda: True
    started = asyncio.Event()

    async def _slow_cycle(*, trigger):
        await f._begin_active_run(trigger=trigger, work_dir="/tmp/parametric-self-test")
        started.set()
        await asyncio.Event().wait()

    f._run_training_cycle_locked = _slow_cycle
    sleep_owner = asyncio.create_task(f.on_post_consolidation({"episodes_created": 1}))
    await asyncio.wait_for(started.wait(), timeout=2)
    sleep_owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sleep_owner
    assert f._active_run is None
    assert f._cycle_in_flight is False
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "interrupted"
    assert "sleep cycle cancelled" in runs[-1]["reason"]


async def test_nightly_exception_is_consumed_and_history_is_failed():
    """An owned nightly task reports failure instead of leaking an unhandled task."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._training_enabled = True
    f._adapter.is_available = lambda: True

    async def _broken_cycle(*, trigger):
        await f._begin_active_run(trigger=trigger, work_dir="/tmp/parametric-self-test")
        raise RuntimeError("synthetic nightly failure")

    f._run_training_cycle_locked = _broken_cycle
    outcome = await f.on_post_consolidation({"episodes_created": 1})
    assert outcome["trained"] is False
    assert "nightly training failed: synthetic nightly failure" == outcome["reason"]
    runs = await f._load_run_history()
    assert runs[-1]["state"] == "failed"
    assert "synthetic nightly failure" in runs[-1]["reason"]


def test_lifecycle_state_backfill_handles_partial_upgrade_shape():
    """Each lifecycle field is independently restored for upgraded instances."""
    f = ParametricSelfFeature.__new__(ParametricSelfFeature)
    f._manual_run_lock = asyncio.Lock()  # present in a partially upgraded object
    f._ensure_training_lifecycle_state()
    assert f._manual_run_generation == 0
    assert f._manual_runs_enabled is True
    assert f._training_shutdown_incomplete is None
    assert f._training_task is None
    assert f._cycle_task is None
    assert f._active_run is None
    assert f._cycle_in_flight is False
    assert f._shutdown_recovery_requires_external_confirmation is False


async def test_status_surfaces_legacy_corpus_cleanup_requirement(tmp_path):
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    legacy = tmp_path / "parametric_self" / "corpus"
    legacy.mkdir(parents=True)
    (legacy / "train.jsonl").write_text('{"text":"legacy"}\n')

    status = await f.parametric_self_status()
    assert status.data["legacy_corpus_cleanup_needed"] is True
    assert "legacy corpus plaintext retained" in status.confirmation


async def test_train_now_cancellation_during_history_reservation_recovers():
    """Cancelling the command while its run record is awaited releases all state."""
    f = await _feature(_FakeStorage(), storage_path="/x/kestrel_prime.db")
    f._adapter.is_available = lambda: True
    entered = asyncio.Event()
    never = asyncio.Event()
    original_append = f._append_run_history

    async def _blocked_append(entry, *, strict=False):
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

    async def _blocked_append(entry, *, strict=False):
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
        _write_run(d, f"Iter 400: Val loss {loss}\n", files=("adapters.safetensors",))
        _write_governed_manifest(d)

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, old)
    _stamp_adapter_receipt(f, new)
    # History records both promotions (made before checkpoint-aware promotion,
    # so they served each run's final weights); new is currently served.
    await f._append_run_history({"promoted": True, "adapter_path": str(old), "trigger": "nightly"})
    await f._append_run_history({"promoted": True, "adapter_path": str(new), "trigger": "nightly"})
    f._active_adapter_path = str(new)

    result = await f.parametric_self_rollback()
    assert result.status == ToolResultStatus.OK
    assert f._active_adapter_path == str(old)
    assert f._active_checkpoint_id == "adapters.safetensors"
    assert f._last_val_loss == pytest.approx(2.900)
    # Rollback is recorded in history.
    runs = await f._load_run_history()
    assert runs[-1]["trigger"] == "rollback"
    assert runs[-1]["checkpoint_id"] == "adapters.safetensors"
    assert runs[-1]["reaches_inference_at_promotion"] is False


async def test_rollback_restores_the_promoted_checkpoint_not_the_final_one(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates"
    old, new = cands / "old111", cands / "new222"
    _write_run(old, JUNE_LOG)
    _write_run(new, IMPROVING_LOG)
    for d in (old, new):
        _write_governed_manifest(d)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, old)
    _stamp_adapter_receipt(f, new)
    await f._append_run_history({
        "promoted": True, "adapter_path": str(old), "checkpoint_id": ITER_200, "val_loss": 1.866,
        "trigger": "nightly",
    })
    await f._append_run_history({
        "promoted": True, "adapter_path": str(new), "checkpoint_id": ITER_400, "val_loss": 1.8,
        "trigger": "nightly",
    })
    f._active_adapter_path, f._active_checkpoint_id = str(new), ITER_400

    result = await f.parametric_self_rollback()
    assert result.status == ToolResultStatus.OK
    assert (f._active_adapter_path, f._active_checkpoint_id) == (str(old), ITER_200)
    assert f._last_val_loss == pytest.approx(1.866)
    assert result.data["served_checkpoint"] == ITER_200


@pytest.mark.parametrize("arg", ["pick99", "adapter_id=pick99"])
async def test_rollback_explicit_adapter_id(tmp_path, arg):
    work = tmp_path / "parametric_self"
    cands = work / "candidates"
    target = cands / "pick99"
    _write_run(target, "Iter 1: Val loss 9.0, x\nIter 200: Val loss 2.750, x\nIter 400: Val loss 2.800, x\n")
    _write_governed_manifest(target)

    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, target)
    f._adapter_lineage[str(target)].update(promoted_checkpoint_id=ITER_200, promoted_val_loss=2.750)
    f._active_adapter_path = "/some/other/served"
    # Accept both the split form ("pick99") and a leaked key=value token.
    result = await f.parametric_self_rollback(adapter_id=arg)
    assert result.status == ToolResultStatus.OK
    assert f._active_adapter_path == str(target)
    assert f._active_checkpoint_id == ITER_200
    assert f._last_val_loss == pytest.approx(2.750)


async def test_rollback_refuses_a_candidate_that_was_never_promoted(tmp_path):
    """Serving an ungated candidate's weights would bypass the fidelity gate."""
    target = tmp_path / "parametric_self" / "candidates" / "never"
    _write_run(target, IMPROVING_LOG)
    _write_governed_manifest(target)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, target)

    result = await f.parametric_self_rollback(adapter_id="never")
    assert result.status == ToolResultStatus.ERROR
    assert "never promoted" in (result.error or "")
    assert f._active_adapter_path is None


async def test_rollback_refuses_a_checkpoint_whose_recorded_loss_is_not_finite(tmp_path):
    target = tmp_path / "parametric_self" / "candidates" / "diverged"
    _write_run(target, IMPROVING_LOG)
    _write_governed_manifest(target)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, target)
    f._adapter_lineage[str(target)].update(promoted_checkpoint_id=ITER_200, promoted_val_loss=math.nan)

    result = await f.parametric_self_rollback(adapter_id="diverged")
    assert result.status == ToolResultStatus.ERROR
    assert "no recorded validation loss" in (result.error or "")
    assert f._active_adapter_path is None


async def test_rollback_takes_the_promoted_score_never_the_training_log(tmp_path):
    """The log's Iter 200 line was measured one step before 0000200 was saved."""
    target = tmp_path / "parametric_self" / "candidates" / "scored"
    _write_run(target, JUNE_LOG)
    _write_governed_manifest(target)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, target)
    f._adapter_lineage[str(target)].update(promoted_checkpoint_id=ITER_200, promoted_val_loss=1.9)

    result = await f.parametric_self_rollback(adapter_id="scored")
    assert result.status == ToolResultStatus.OK
    assert f._last_val_loss == 1.9


async def test_rollback_refuses_a_quarantined_adapter_without_reverifying_it(tmp_path):
    target = tmp_path / "parametric_self" / "candidates" / "bad"
    _write_run(target, IMPROVING_LOG)
    _write_governed_manifest(target)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, target)
    f._adapter_lineage[str(target)]["promoted_checkpoint_id"] = ITER_200
    f._quarantined_adapters[str(target)] = "governed assertion lineage invalidated; rebuild required"

    async def _must_not_verify(*_args, **_kwargs):
        raise AssertionError("a quarantined adapter is not re-verified")

    f._verify_adapter_lineage = _must_not_verify
    result = await f.parametric_self_rollback(adapter_id="bad")
    assert result.status == ToolResultStatus.ERROR
    assert "quarantined" in (result.error or "")
    assert f._active_adapter_path is None


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
    _write_run(cands, "Iter 10: training started...\n")  # no Val loss line
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    await f._append_run_history({"promoted": True, "adapter_path": str(cands), "checkpoint_id": ITER_200})
    result = await f.parametric_self_rollback(adapter_id="incomplete")
    assert result.status == ToolResultStatus.ERROR
    assert "no recorded validation loss" in (result.error or "")
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


@pytest.mark.parametrize("log", [JUNE_LOG, JUNE_LOG_MLX_032], ids=["mlx-lm-0.31", "mlx-lm-0.32"])
async def test_recovery_surface_offers_what_adopt_promotes_from_recorded_scores(tmp_path, log):
    """Whatever the training log looks like, status lists the candidate adopt
    would promote, at the score its iter-200 file recorded, not the final one."""
    cands = tmp_path / "parametric_self" / "candidates" / "june"
    _write_run(cands, log)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, JUNE_SCORES)
    await f.set_config({"gate": {"overfit_ratio": 1.5}})

    assert (await f.parametric_self_status()).data["recoverable_adapters"] == ["june"]
    listing = await f.parametric_self_adapters()
    assert listing.data["recoverable_adapters"] == ["june"]
    listed = listing.data["adapters"][0]
    assert listed["recoverable"] is True
    assert (listed["checkpoint_id"], listed["val_loss"]) == (ITER_200, 1.866)
    assert (listed["final_checkpoint_id"], listed["final_val_loss"]) == (ITER_400, 2.688)
    assert listed["adoption"].startswith("fidelity gate promote (passed)")

    adopted = await f.parametric_self_adopt(adapter_id="june")
    assert adopted.status == ToolResultStatus.OK
    assert (adopted.data["served_checkpoint"], adopted.data["served_val_loss"]) == (ITER_200, 1.866)
    served = (await f.parametric_self_adapters()).data["adapters"][0]
    assert served["served"] is True and served["recoverable"] is False
    assert (served["checkpoint_id"], served["val_loss"]) == (ITER_200, 1.866)


@pytest.mark.parametrize("log", [JUNE_LOG, JUNE_LOG_MLX_032], ids=["mlx-lm-0.31", "mlx-lm-0.32"])
async def test_recovery_surface_withholds_a_candidate_adopt_rejects(tmp_path, log):
    """Under the default gate the overfit June run is shown with its recorded
    scores and why adoption would reject it, and is not offered as recoverable."""
    cands = tmp_path / "parametric_self" / "candidates" / "june"
    _write_run(cands, log)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, JUNE_SCORES)

    assert (await f.parametric_self_status()).data["recoverable_adapters"] == []
    listed = (await f.parametric_self_adapters()).data["adapters"][0]
    assert listed["recoverable"] is False
    assert (listed["checkpoint_id"], listed["val_loss"]) == (ITER_200, 1.866)
    assert (listed["final_checkpoint_id"], listed["final_val_loss"]) == (ITER_400, 2.688)
    assert listed["adoption"].startswith("fidelity gate reject (overfit)")

    adopted = await f.parametric_self_adopt(adapter_id="june")
    assert adopted.status == ToolResultStatus.ERROR
    assert "fidelity gate reject (overfit)" in (adopted.error or "")


async def test_recovery_surface_shows_no_score_for_a_candidate_without_a_gate_record(tmp_path):
    """A logged loss is not a score: with no gate record nothing is shown or offered."""
    cands = tmp_path / "parametric_self" / "candidates" / "legacy"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)

    listed = (await f.parametric_self_adapters()).data["adapters"][0]
    assert listed["recoverable"] is False
    assert listed["checkpoint_id"] is None and listed["val_loss"] is None
    assert listed["final_checkpoint_id"] is None and listed["final_val_loss"] is None
    assert listed["adoption"].startswith("fidelity gate inconclusive (val_set_missing)")
    assert (await f.parametric_self_status()).data["recoverable_adapters"] == []


async def test_recovery_surface_withholds_a_candidate_whose_recorded_checkpoint_is_gone(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "pruned"
    _write_run(cands, IMPROVING_LOG, files=(ITER_200,))
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands)

    listed = (await f.parametric_self_adapters()).data["adapters"][0]
    assert listed["recoverable"] is False
    assert listed["adoption"] == f"refused: checkpoint {ITER_400} is no longer on disk"
    assert (await f.parametric_self_status()).data["recoverable_adapters"] == []


async def test_recovery_surface_never_redecides_a_quarantined_candidate(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "bad"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands)
    f._quarantined_adapters[str(cands)] = "governed assertion lineage invalidated; rebuild required"

    class _NoScoring(FidelityGate):
        def evaluate(self, **_kwargs):
            raise AssertionError("a quarantined candidate is not re-decided")

    f._gate = _NoScoring()
    listed = (await f.parametric_self_adapters()).data["adapters"][0]
    assert listed["recoverable"] is False
    assert listed["adoption"] is None
    assert listed["quarantined_reason"] == "governed assertion lineage invalidated; rebuild required"
    assert (listed["checkpoint_id"], listed["val_loss"]) == (ITER_400, 1.800)
    assert (await f.parametric_self_status()).data["recoverable_adapters"] == []


async def test_adopt_persists_served_and_appends_adopt_history(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    storage = _FakeStorage()
    f = await _feature(storage, storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands)

    result = await f.parametric_self_adopt(adapter_id="ded33cd017d9")
    assert result.status == ToolResultStatus.OK
    assert f._active_adapter_path == str(cands)
    assert f._active_checkpoint_id == ITER_400
    assert f._last_val_loss == pytest.approx(1.800)
    assert f._config_node_id() in storage.nodes  # served pointer persisted
    runs = await f._load_run_history()
    assert runs[-1]["trigger"] == "adopt"
    assert runs[-1]["adapter_path"] == str(cands)
    assert runs[-1]["promoted"] is True
    assert runs[-1]["checkpoint_id"] == ITER_400
    assert runs[-1]["reaches_inference_at_promotion"] is False
    gate = runs[-1]["gate"]
    assert gate["decision"] == "promote"
    assert gate["baseline_source"] == "base" and gate["baseline_val_loss"] == 8.977
    assert gate["val_set_digest"] == "sha256:" + "cd" * 32


async def test_adopt_promotes_the_best_checkpoint_when_overfit_check_is_relaxed(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    _write_run(cands, JUNE_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, JUNE_SCORES)  # rejected for overfit when trained
    await f.set_config({"gate": {"overfit_ratio": 1.5}})

    result = await f.parametric_self_adopt(adapter_id="ded33cd017d9")
    assert result.status == ToolResultStatus.OK
    assert f._active_checkpoint_id == ITER_200


@pytest.mark.parametrize("gate_config", [
    {}, {"overfit_check": False}, {"overfit_ratio": 1.5},
], ids=["default", "overfit_check_off", "ratio_1.50"])
@pytest.mark.parametrize("scores", [
    {ITER_200: None, ITER_400: 2.688}, {ITER_200: 1.866, ITER_400: None},
], ids=["intermediate_missing", "final_missing"])
async def test_adopt_with_a_missing_recorded_score_is_inconclusive_and_unhealthy(tmp_path, gate_config, scores):
    """A June record that lost one checkpoint's score is never adopted, under any gate config:
    the unscored file could have been the best, so the scored one must not win by default."""
    cands = tmp_path / "parametric_self" / "candidates" / "june"
    _write_run(cands, JUNE_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, scores)
    if gate_config:
        await f.set_config({"gate": gate_config})

    status = await f.parametric_self_status()
    assert status.data["recoverable_adapters"] == []
    listed = (await f.parametric_self_adapters()).data["adapters"][0]
    assert listed["recoverable"] is False
    assert listed["adoption"].startswith("fidelity gate inconclusive (candidate_loss_missing)")

    result = await f.parametric_self_adopt(adapter_id="june")
    assert result.status == ToolResultStatus.ERROR
    assert "fidelity gate inconclusive (candidate_loss_missing)" in (result.error or "")
    assert f._active_adapter_path is None and f._active_checkpoint_id is None
    runs = await f._load_run_history()
    assert runs[-1]["trigger"] == "adopt" and runs[-1]["promoted"] is False
    gate = runs[-1]["gate"]
    assert (gate["decision"], gate["reason"]) == ("inconclusive", "candidate_loss_missing")
    assert gate["best_checkpoint_id"] is None and gate["promoted_checkpoint_id"] is None
    assert gate["ratio"] is None

    status = (await f.parametric_self_status()).data
    assert status["served_adapter"] is None
    assert status["fidelity_gate"]["last_decision"] == gate
    assert status["fidelity_gate"]["healthy"] is False


async def test_adopt_without_a_recorded_val_set_is_inconclusive(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "legacy"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)

    result = await f.parametric_self_adopt(adapter_id="legacy")
    assert result.status == ToolResultStatus.ERROR
    assert "inconclusive (val_set_missing)" in (result.error or "")
    assert f._active_adapter_path is None
    runs = await f._load_run_history()
    assert runs[-1]["gate"]["decision"] == "inconclusive"


async def test_adopt_with_an_adapter_served_is_inconclusive_not_a_stored_comparison(tmp_path):
    """The served adapter cannot be scored on a deleted val set; no stored loss stands in."""
    cands = tmp_path / "parametric_self" / "candidates" / "next"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands)
    served = tmp_path / "parametric_self" / "candidates" / "old"
    _serve_governed(f, served)
    f._last_val_loss = 50.0  # a stored loss that would let the candidate through

    result = await f.parametric_self_adopt(adapter_id="next")
    assert result.status == ToolResultStatus.ERROR
    assert "inconclusive (baseline_changed)" in (result.error or "")
    assert "scored against base" in (result.error or "")
    assert f._active_adapter_path == str(served)
    assert str(served) not in f._quarantined_adapters


async def test_adopt_against_a_served_adapter_whose_lineage_fails_is_inconclusive(tmp_path):
    """The served adapter is the baseline only while its governed lineage holds.

    Here its source assertion has left the governed snapshot. Its recorded
    score would let the candidate through; instead the served adapter is
    quarantined and the decision is inconclusive, with no fallback to the base.
    """
    cands = tmp_path / "parametric_self" / "candidates" / "next"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, baseline=(f"served:old/{ITER_200}", 2.5))
    served = tmp_path / "parametric_self" / "candidates" / "old"
    _serve_governed(f, served, assertion_id="assertion:erased")

    result = await f.parametric_self_adopt(adapter_id="next")
    assert result.status == ToolResultStatus.ERROR
    assert "inconclusive (baseline_changed)" in (result.error or "")
    assert "failed its governed-lineage check" in (result.error or "")
    assert f._active_adapter_path is None
    assert f._quarantined_adapters[str(served)] == (
        "governed assertion lineage no longer current; rebuild required"
    )
    entry = (await f._load_run_history())[-1]
    assert (entry["trigger"], entry["promoted"], entry["gate_record_persisted"]) == ("adopt", False, True)
    gate = entry["gate"]
    assert (gate["decision"], gate["reason"]) == ("inconclusive", "baseline_changed")
    assert gate["baseline_source"] == f"served:old/{ITER_200}"
    assert gate["baseline_val_loss"] is None


async def test_adopt_after_the_base_model_changed_is_inconclusive(tmp_path):
    """The recorded scores were measured on another base model; they do not carry over."""
    cands = tmp_path / "parametric_self" / "candidates" / "next"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands)
    await f.set_config({"base_model": "mlx-community/another-base-4bit"})

    result = await f.parametric_self_adopt(adapter_id="next")
    assert result.status == ToolResultStatus.ERROR
    assert "inconclusive (baseline_changed)" in (result.error or "")
    assert "mlx-community/another-base-4bit" in (result.error or "")
    assert f._active_adapter_path is None
    gate = (await f._load_run_history())[-1]["gate"]
    assert gate["decision"] == "inconclusive"
    assert gate["base_model"] == TextLoRAConfig().base_model


async def test_adopt_refuses_a_record_that_names_no_base_model(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "next"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, base_model=None)

    result = await f.parametric_self_adopt(adapter_id="next")
    assert result.status == ToolResultStatus.ERROR
    assert "inconclusive (baseline_changed)" in (result.error or "")
    assert f._active_adapter_path is None


async def test_adopt_uses_a_served_baseline_scored_in_the_candidates_own_gate_run(tmp_path):
    """While the served checkpoint is unchanged, its training-run score still stands."""
    cands = tmp_path / "parametric_self" / "candidates" / "next"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, baseline=(f"served:old/{ITER_200}", 2.5))
    _serve_governed(f, tmp_path / "parametric_self" / "candidates" / "old")

    result = await f.parametric_self_adopt(adapter_id="next")
    assert result.status == ToolResultStatus.OK
    assert result.data["gate"]["baseline_val_loss"] == 2.5
    assert f._active_checkpoint_id == ITER_400


async def test_adopt_decides_from_recorded_scores_never_the_training_log(tmp_path):
    """The log credits 0000400 with 1.0; the file itself scored 10.0 when it was trained."""
    cands = tmp_path / "parametric_self" / "candidates" / "logged"
    _write_run(cands, "Iter 1: Val loss 9.0, x\nIter 200: Val loss 1.866, x\nIter 400: Val loss 1.000, x\n")
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, {ITER_200: 1.866, ITER_400: 10.0})

    result = await f.parametric_self_adopt(adapter_id="logged")
    assert result.status == ToolResultStatus.ERROR
    assert "fidelity gate reject (overfit)" in (result.error or "")
    assert f._active_adapter_path is None


async def test_adopt_refuses_a_malformed_training_gate_record(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "garbled"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await f._append_run_history({
        "run_id": "train-garbled", "trigger": "nightly", "adapter_path": str(cands),
        "gate": {"decision": "promote", "val_set_digest": "sha256:x", "val_set_examples": 7},
    })

    result = await f.parametric_self_adopt(adapter_id="garbled")
    assert result.status == ToolResultStatus.ERROR
    assert "malformed fidelity-gate record" in (result.error or "")
    assert f._active_adapter_path is None


@pytest.mark.parametrize("arg", ["ded33cd017d9", "adapter_id=ded33cd017d9"])
async def test_adopt_accepts_leaked_key_value_token(tmp_path, arg):
    cands = tmp_path / "parametric_self" / "candidates" / "ded33cd017d9"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands)
    result = await f.parametric_self_adopt(adapter_id=arg)
    assert result.status == ToolResultStatus.OK
    assert f._active_adapter_path == str(cands)


async def test_adopt_refuses_a_recorded_checkpoint_that_is_gone_from_disk(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "pruned"
    _write_run(cands, IMPROVING_LOG, files=(ITER_200,))
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands)
    result = await f.parametric_self_adopt(adapter_id="pruned")
    assert result.status == ToolResultStatus.ERROR
    assert f"checkpoint {ITER_400} is no longer on disk" in (result.error or "")
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
    """Emma's overfit June run must not be adopted under the default gate."""
    cands = tmp_path / "parametric_self" / "candidates" / "toobad"
    _write_run(cands, JUNE_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands, JUNE_SCORES)
    result = await f.parametric_self_adopt(adapter_id="toobad")
    assert result.status == ToolResultStatus.ERROR
    assert "fidelity gate reject (overfit)" in (result.error or "")
    assert f._active_adapter_path is None


async def test_adopt_refuses_a_quarantined_candidate_without_rescoring_it(tmp_path):
    cands = tmp_path / "parametric_self" / "candidates" / "bad"
    _write_run(cands, IMPROVING_LOG)
    _write_governed_manifest(cands)
    f = await _feature(_FakeStorage(), storage_path=str(tmp_path / "kestrel_prime.db"))
    _stamp_adapter_receipt(f, cands)
    await _record_training_gate(f, cands)
    f._quarantined_adapters[str(cands)] = "governed assertion lineage invalidated; rebuild required"
    history_before = json.dumps(await f._load_run_history(), sort_keys=True)

    class _NoScoring:
        def evaluate(self, **_kwargs):
            raise AssertionError("a quarantined candidate is not re-scored")

    f._gate = _NoScoring()
    result = await f.parametric_self_adopt(adapter_id="bad")
    assert result.status == ToolResultStatus.ERROR
    assert "quarantined" in (result.error or "")
    assert f._active_adapter_path is None
    assert json.dumps(await f._load_run_history(), sort_keys=True) == history_before


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
    assert by_id["abc123def456"]["val_loss"] is None  # nothing scored yet
    assert by_id["abc123def456"]["adoption"] is None
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
