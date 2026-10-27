"""The fidelity gate wired through the feature: records, promotion, health (#30).

Runs the real ``run_nightly_cycle`` and promotion path with a fake trainer that
writes mlx_lm's files and scores each one, against a host with the full
governed-corpus surface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kestrel_sovereign.agent.sleep import SleepHookStatus, SleepMixin
from kestrel_sovereign.features.training.types import TrainingState, TrainingStatus

from kestrel_feature_parametric_self import ParametricSelfFeature, ParametricSelfSleepHook
from kestrel_feature_parametric_self.fidelity import GATE_RECORD_FIELDS

ITER_200 = "0000200_adapters.safetensors"
ITER_400 = "0000400_adapters.safetensors"
WEIGHT_FILES = ("0000100_adapters.safetensors", ITER_200, "0000300_adapters.safetensors", ITER_400, "adapters.safetensors")
# What each set of weights scores on the run's validation set: the base, and
# the checkpoint files the run saved. Emma's June run (#11): base 8.977,
# iter 200 1.866, final iter 400 2.688.
JUNE = {"base": 8.977, ITER_200: 1.866, ITER_400: 2.688}
IMPROVING = {"base": 9.0, ITER_200: 1.5, ITER_400: 1.4}
IMPROVING_LOG = "Iter 1: Val loss 9.000, x\nIter 200: Val loss 1.500, x\nIter 400: Val loss 1.400, x\n"


def _snapshot():
    assertion = SimpleNamespace(
        assertion_id="assertion:one", revision_id="revision:one",
        subject=SimpleNamespace(value="https://example.test/agent"),
        predicate=SimpleNamespace(value="https://example.test/lesson"),
        object=SimpleNamespace(lexical_form="governed lesson"),
    )
    return SimpleNamespace(
        verified=True,
        examples=(SimpleNamespace(
            assertion=assertion, content_hash="sha256:revision:one", source_occurrences=(),
            decision=SimpleNamespace(included=True, reason=SimpleNamespace(value="included")),
        ),),
        snapshot_hash="sha256:snapshot:one", tenant_id="tenant:test",
        policy=SimpleNamespace(digest="sha256:policy"),
        checkpoint=SimpleNamespace(tenant_id="tenant:test", generation=4, latest_event_id="event:4"),
        capability_versions={"semantic_maintenance": "1"},
    )


class _Host:
    """Agent storage: graph nodes plus the governed corpus capability.

    Nodes are copied in and out, as a serializing store would, so a write
    that fails leaves nothing behind. ``fail_gate_record_writes`` fails only a
    history write that adds a gate record; ``fail_history_writes`` and
    ``fail_history_reads`` fail every history write or read.
    """

    def __init__(self):
        self.snapshot = _snapshot()
        self.nodes = {}
        self.fail_gate_record_writes = False
        self.fail_history_writes = False
        self.fail_history_reads = False

    async def governed_assertion_corpus_snapshot(self, **_kwargs):
        return self.snapshot

    async def governed_assertion_corpus_changes_since(self, base, **_kwargs):
        return SimpleNamespace(
            tombstones=(), since_checkpoint=base.checkpoint, checkpoint=self.snapshot.checkpoint,
            snapshot_hash="sha256:delta", observability=SimpleNamespace(policy_digest="sha256:policy"),
        )

    @staticmethod
    def _gates(node):
        return sum("gate" in run for run in node.properties["runs"]) if node is not None else 0

    async def add_node(self, node):
        if node.node_id.startswith("parametric_self_runs:"):
            if self.fail_history_writes:
                raise OSError("history store unavailable")
            if self.fail_gate_record_writes and self._gates(node) > self._gates(self.nodes.get(node.node_id)):
                raise OSError("history store unavailable")
        self.nodes[node.node_id] = copy.deepcopy(node)

    async def get_node(self, node_id):
        if self.fail_history_reads and node_id.startswith("parametric_self_runs:"):
            raise OSError("history store read timed out")
        return copy.deepcopy(self.nodes.get(node_id))


class _Trainer:
    """Writes the candidate's mlx_lm files; scores the weights a view links to."""

    def __init__(self, scores=IMPROVING, served_score=5.0, served_error=None):
        self.scores, self.served_score, self.served_error = scores, served_score, served_error
        self.trained_val_bytes = None
        self.scored_test_bytes = []
        self._candidate = None
        self._logs = {}

    def is_available(self):
        return True

    async def start_training(self, agent_id, config):
        self.trained_val_bytes = (Path(config.data_dir) / "valid.jsonl").read_bytes()
        candidate = Path(config.adapter_path)
        self._candidate = candidate.resolve()
        for name in [*self.scores, "adapters.safetensors"]:
            if name != "base":
                (candidate / name).write_bytes(b"weights")
        (candidate / "adapter_config.json").write_text("{}")
        return TrainingStatus(job_id="train", state=TrainingState.COMPLETED, progress=1.0)

    async def start_evaluation(self, agent_id, config, *, log_dir):
        if config.adapter_path is None:
            loss = self.scores["base"]
        else:
            weights = (Path(config.adapter_path) / "adapters.safetensors").resolve()
            if weights.parent != self._candidate:
                if self.served_error is not None:
                    raise self.served_error
                loss = self.served_score
            else:
                loss = self.scores[weights.name]
        self.scored_test_bytes.append((Path(config.data_dir) / "test.jsonl").read_bytes())
        job_id = f"eval-{len(self._logs)}"
        self._logs[job_id] = f"Testing\nTest loss {loss:.3f}, Test ppl 1.0.\n"
        return TrainingStatus(job_id=job_id, state=TrainingState.COMPLETED, progress=1.0)

    async def get_status(self, job_id):
        return TrainingStatus(job_id=job_id, state=TrainingState.COMPLETED, progress=1.0)

    def read_training_log(self, job_id):
        return self._logs.get(job_id, "")


def _reflections(tmp_path, rows=5) -> str:
    """Grounded insights; with the governed example they give a validation row."""
    db = str(tmp_path / "cog.db")
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE reflection_insights (id TEXT, type TEXT, title TEXT NOT NULL, "
        "description TEXT, suggested_action TEXT)"
    )
    con.executemany(
        "INSERT INTO reflection_insights VALUES (?,?,?,?,?)",
        [(str(i), "failure", f"Lesson {i}", f"Do thing {i}.", "") for i in range(rows)],
    )
    con.commit()
    con.close()
    return db


async def _feature(tmp_path, trainer=None, *, rows=5):
    host = _Host()
    agent = MagicMock()
    agent.storage = host
    agent.storage_path = _reflections(tmp_path, rows)
    agent.parametric_self_work_dir = str(tmp_path / "work")
    agent.parametric_self_governed_corpus_policy = SimpleNamespace(digest="sha256:policy")
    agent.semantic_inference_profile = None
    agent.is_test_instance = False
    agent.agent_id = "emma"
    agent.sleep_hooks = []
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    feature._adapter = trainer or _Trainer()
    agent.get_feature = MagicMock(
        side_effect=lambda name: feature if name in ("parametric_self", "ParametricSelfFeature") else None
    )
    return feature, host


async def _last_run(feature):
    return (await feature._load_run_history())[-1]


def _persisted_config(host, feature):
    node = host.nodes.get(feature._config_node_id())
    return node.properties["config"] if node is not None else {}


def _serve_existing(tmp_path, feature, name="servedrun", checkpoint=ITER_200):
    served = tmp_path / "work" / "candidates" / name
    served.mkdir(parents=True)
    (served / checkpoint).write_bytes(b"served weights")
    (served / "adapter_config.json").write_text("{}")
    feature._active_adapter_path, feature._active_checkpoint_id = str(served), checkpoint
    return served


# ----------------------------------------------------------------------
# Decisions reach history before anything is promoted
# ----------------------------------------------------------------------

async def test_june_run_is_rejected_for_overfit_and_the_comparison_is_recorded(tmp_path):
    feature, host = await _feature(tmp_path, _Trainer(JUNE))

    outcome = await feature._run_training_cycle_locked(trigger="nightly")

    assert outcome["trained"] is True and outcome["promoted"] is False
    assert outcome["gate_record_persisted"] is True
    entry = await _last_run(feature)
    gate = entry["gate"]
    assert tuple(gate) == GATE_RECORD_FIELDS
    assert (gate["decision"], gate["reason"]) == ("reject", "overfit")
    assert (gate["best_checkpoint_id"], gate["best_val_loss"]) == (ITER_200, 1.866)
    assert (gate["final_checkpoint_id"], gate["final_val_loss"]) == (ITER_400, 2.688)
    assert (gate["baseline_source"], gate["baseline_val_loss"]) == ("base", 8.977)
    assert gate["promoted_checkpoint_id"] is None
    assert feature._active_adapter_path is None
    status = await feature.parametric_self_status()
    assert status.data["fidelity_gate"]["last_decision"] == gate
    assert status.data["fidelity_gate"]["last_decision_persisted"] is True
    assert status.data["fidelity_gate"]["healthy"] is True
    assert "Last fidelity-gate decision: reject (overfit)" in status.confirmation


async def test_improving_run_promotes_its_best_checkpoint_and_persists_it(tmp_path):
    feature, host = await _feature(tmp_path)

    outcome = await feature._run_training_cycle_locked(trigger="nightly")

    assert outcome["promoted"] is True and outcome["checkpoint_id"] == ITER_400
    assert feature._active_checkpoint_id == ITER_400
    assert feature._last_val_loss == 1.4
    config = _persisted_config(host, feature)
    assert config["active_adapter_path"] == feature._active_adapter_path
    assert config["active_checkpoint_id"] == ITER_400
    lineage = feature._adapter_lineage[feature._active_adapter_path]
    assert lineage["state"] == "served" and lineage["promoted_checkpoint_id"] == ITER_400
    assert lineage["promoted_val_loss"] == 1.4
    entry = await _last_run(feature)
    assert entry["checkpoint_id"] == ITER_400 and entry["gate"]["decision"] == "promote"

    restarted = ParametricSelfFeature(agent=feature.agent)
    await restarted.initialize()
    await restarted._restore_persisted_config()
    assert restarted._active_checkpoint_id == ITER_400


async def test_failure_after_the_record_write_promotes_nothing(tmp_path):
    """R9: the record exists, and the injected failure precedes every side effect."""
    feature, host = await _feature(tmp_path)

    async def _fail_before_side_effect(*_args, **_kwargs):
        raise RuntimeError("injected failure before promotion")

    feature._apply_promotion = _fail_before_side_effect
    with pytest.raises(RuntimeError, match="injected failure"):
        await feature._run_training_cycle(trigger="nightly")

    entry = await _last_run(feature)
    assert entry["gate"]["decision"] == "promote"
    assert entry["gate"]["promoted_checkpoint_id"] == ITER_400
    assert entry["state"] == "failed"
    assert feature._active_adapter_path is None and feature._active_checkpoint_id is None
    assert _persisted_config(host, feature).get("active_adapter_path") is None


async def test_a_failed_record_write_promotes_nothing(tmp_path):
    """R9 reverse: if the decision cannot be recorded, it is not acted on."""
    feature, host = await _feature(tmp_path)
    host.fail_gate_record_writes = True
    promotions = []

    async def _spy(*args, **_kwargs):
        promotions.append(args)

    feature._apply_promotion = _spy
    feature._training_enabled = True
    outcome = await ParametricSelfSleepHook(feature).on_post_consolidation(feature.agent, {})

    assert promotions == []
    assert outcome["promoted"] is False
    assert outcome["gate_record_persisted"] is False
    assert outcome["gate"]["decision"] == "promote"  # the decision, not acted on
    assert "could not be persisted" in outcome["reason"]
    assert feature._active_adapter_path is None
    assert SleepMixin._hook_outcome(outcome)[0] is SleepHookStatus.FAILED
    assert "gate" not in await _last_run(feature)


# ----------------------------------------------------------------------
# Inconclusive is never healthy (R4)
# ----------------------------------------------------------------------

async def test_inconclusive_run_is_not_healthy_on_status_or_the_sleep_hook(tmp_path):
    feature, _host = await _feature(tmp_path, rows=0)  # one example; empty validation split
    feature._training_enabled = True

    outcome = await ParametricSelfSleepHook(feature).on_post_consolidation(feature.agent, {})

    assert outcome["gate"]["decision"] == "inconclusive"
    assert outcome["gate"]["reason"] == "val_set_empty"
    assert outcome["promoted"] is False
    assert SleepMixin._hook_outcome(outcome)[0] is SleepHookStatus.FAILED
    status = await feature.parametric_self_status()
    assert status.data["fidelity_gate"]["healthy"] is False
    assert "WARNING: the last fidelity-gate decision was inconclusive (val_set_empty)" in status.confirmation


async def test_no_gate_decision_yet_is_not_reported_healthy(tmp_path):
    feature, _host = await _feature(tmp_path)
    status = await feature.parametric_self_status()
    assert status.data["fidelity_gate"]["last_decision"] is None
    assert status.data["fidelity_gate"]["healthy"] is False


async def _promote_then_attempt_an_inconclusive_run(feature, host, **failures):
    """A promoted run, then an inconclusive one whose gate record write fails."""
    promoted = await feature._run_training_cycle_locked(trigger="nightly")
    assert promoted["promoted"] is True
    feature._adapter = _Trainer(served_error=RuntimeError("metal device lost"))
    for name, value in failures.items():
        setattr(host, name, value)
    outcome = await feature._run_training_cycle_locked(trigger="nightly")
    assert outcome["gate"]["decision"] == "inconclusive"
    assert outcome["gate_record_persisted"] is False
    for name in failures:
        setattr(host, name, False)
    return promoted["gate"], outcome


async def test_an_unrecorded_attempt_after_a_promotion_is_not_healthy(tmp_path):
    """The earlier promotion is history; it never makes the newer attempt healthy."""
    feature, host = await _feature(tmp_path)
    promotion, outcome = await _promote_then_attempt_an_inconclusive_run(
        feature, host, fail_gate_record_writes=True,
    )
    assert (await _last_run(feature))["gate_record_persisted"] is False

    status = (await feature.parametric_self_status()).data["fidelity_gate"]
    assert status["healthy"] is False
    assert status["last_decision"] == outcome["gate"]
    assert status["last_decision_persisted"] is False
    assert status["last_recorded_decision"] == promotion

    # After a restart only history witnesses the attempt: its terminal entry.
    restarted = ParametricSelfFeature(agent=feature.agent)
    await restarted.initialize()
    restarted_status = await restarted.parametric_self_status()
    gate = restarted_status.data["fidelity_gate"]
    assert gate["healthy"] is False
    assert gate["last_decision"] is None and gate["last_decision_persisted"] is False
    assert gate["last_recorded_decision"] == promotion
    assert "could not be recorded" in restarted_status.confirmation
    assert "The last recorded decision was promote (passed)" in restarted_status.confirmation


async def test_an_attempt_that_left_no_trace_in_history_is_still_not_healthy(tmp_path):
    """Neither the gate record nor the terminal entry landed; this process still knows."""
    feature, host = await _feature(tmp_path)
    promotion, outcome = await _promote_then_attempt_an_inconclusive_run(
        feature, host, fail_history_writes=True,
    )
    assert (await _last_run(feature))["state"] == "completed"
    assert (await _last_run(feature))["gate"] == promotion  # history only shows the promotion

    status = (await feature.parametric_self_status()).data["fidelity_gate"]
    assert status["healthy"] is False
    assert status["last_decision"] == outcome["gate"]
    assert status["last_decision_persisted"] is False


# ----------------------------------------------------------------------
# Same-run baseline (R5)
# ----------------------------------------------------------------------

async def test_record_digest_is_the_val_set_both_scores_used(tmp_path):
    trainer = _Trainer(served_score=5.0)
    feature, _host = await _feature(tmp_path, trainer)
    _serve_existing(tmp_path, feature)

    await feature._run_training_cycle_locked(trigger="nightly")

    gate = (await _last_run(feature))["gate"]
    assert gate["val_set_digest"] == "sha256:" + hashlib.sha256(trainer.trained_val_bytes).hexdigest()
    # Both candidate checkpoints and the served adapter, on that one set.
    assert trainer.scored_test_bytes == [trainer.trained_val_bytes] * 3
    assert gate["baseline_source"] == f"served:servedrun/{ITER_200}"


@pytest.mark.parametrize("served", [True, False], ids=["served-baseline", "base-baseline"])
async def test_a_stale_stored_baseline_loss_is_ignored(tmp_path, served):
    """A stored 1.0 would reject the 1.4 candidate on margin; the gate never reads it."""
    feature, _host = await _feature(tmp_path, _Trainer(served_score=5.0))
    if served:
        _serve_existing(tmp_path, feature)
    feature._last_val_loss = 1.0

    outcome = await feature._run_training_cycle_locked(trigger="nightly")

    assert outcome["promoted"] is True
    assert outcome["gate"]["baseline_val_loss"] == (5.0 if served else 9.0)


# ----------------------------------------------------------------------
# Quarantined adapters are left alone (R8)
# ----------------------------------------------------------------------

async def test_a_quarantined_adapter_is_not_rescored_promoted_or_rewritten(tmp_path):
    feature, host = await _feature(tmp_path)
    quarantined = str(tmp_path / "work" / "candidates" / "quarantined")
    _write = Path(quarantined)
    _write.mkdir(parents=True)
    for name in WEIGHT_FILES:
        (_write / name).write_bytes(b"old weights")
    (_write / "train.log").write_text(IMPROVING_LOG)
    feature._quarantined_adapters[quarantined] = "governed assertion lineage invalidated; rebuild required"
    feature._adapter_lineage[quarantined] = {
        "state": "invalid", "invalidation_reason": "governed assertion lineage invalidated; rebuild required",
    }
    await feature._append_run_history({
        "run_id": "oldrun", "trigger": "nightly", "state": "completed", "trained": True,
        "promoted": False, "adapter_path": quarantined,
        "gate": {"decision": "promote", "promoted_checkpoint_id": ITER_400},
    })
    await feature._persist_config()
    entry_before = json.dumps((await feature._load_run_history())[0], sort_keys=True)
    lineage_before = json.dumps(_persisted_config(host, feature)["adapter_lineage"][quarantined], sort_keys=True)

    outcome = await feature._run_training_cycle_locked(trigger="nightly")

    assert outcome["promoted"] is True
    assert feature._active_adapter_path != quarantined
    assert outcome["gate"]["baseline_source"] == "base"  # never the quarantined adapter
    runs = await feature._load_run_history()
    assert json.dumps(runs[0], sort_keys=True) == entry_before
    lineage_after = _persisted_config(host, feature)["adapter_lineage"][quarantined]
    assert json.dumps(lineage_after, sort_keys=True) == lineage_before
    assert feature._quarantined_adapters[quarantined].startswith("governed assertion lineage")

    adopt = await feature.parametric_self_adopt("quarantined")
    assert adopt.status.value == "error" and "quarantined" in (adopt.error or "")
    assert json.dumps((await feature._load_run_history())[0], sort_keys=True) == entry_before


# ----------------------------------------------------------------------
# reaches_inference is recorded from the effective switch (#31)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("reaches", [False, True], ids=["switch-off", "modelled-on"])
async def test_history_and_lineage_record_reaches_inference_at_train_and_promotion(tmp_path, reaches):
    """``off`` is the only value today, so "on" is modelled at the one accessor."""
    feature, _host = await _feature(tmp_path)
    if reaches:
        feature.adapter_for_inference = lambda: "/modelled/live/adapter"

    outcome = await feature._run_training_cycle_locked(trigger="nightly")

    assert outcome["promoted"] is True
    entry = await _last_run(feature)
    assert entry["reaches_inference_at_train"] is reaches
    assert entry["reaches_inference_at_promotion"] is reaches
    lineage = feature._adapter_lineage[feature._active_adapter_path]
    assert lineage["reaches_inference_at_train"] is reaches
    assert lineage["reaches_inference_at_promotion"] is reaches


async def test_unpromoted_run_records_no_promotion_time_value(tmp_path):
    feature, _host = await _feature(tmp_path, _Trainer(JUNE))
    await feature._run_training_cycle_locked(trigger="nightly")
    entry = await _last_run(feature)
    assert entry["reaches_inference_at_train"] is False
    assert entry["reaches_inference_at_promotion"] is None


# ----------------------------------------------------------------------
# gate.* config
# ----------------------------------------------------------------------

async def test_gate_config_is_overridable_persisted_and_reported(tmp_path):
    feature, host = await _feature(tmp_path, _Trainer(JUNE))
    await feature.set_config({"gate": {"overfit_ratio": 1.5}})
    assert (await feature.get_config())["gate"] == {
        "margin": 0.02, "overfit_ratio": 1.5, "overfit_check": True,
    }
    assert _persisted_config(host, feature)["gate"]["overfit_ratio"] == 1.5

    outcome = await feature._run_training_cycle_locked(trigger="nightly")
    assert outcome["checkpoint_id"] == ITER_200

    restarted = ParametricSelfFeature(agent=feature.agent)
    await restarted.initialize()
    await restarted._restore_persisted_config()
    assert restarted._gate.overfit_ratio == 1.5


async def test_invalid_gate_config_is_refused_and_applies_nothing(tmp_path):
    feature, _host = await _feature(tmp_path)
    with pytest.raises(ValueError):
        await feature.set_config({"gate": {"margin": -0.5}, "enable_nightly_training": True})
    assert feature._gate.margin == 0.02
    assert feature._training_enabled is False


async def test_invalid_persisted_gate_config_keeps_pinned_defaults_and_is_reported(tmp_path):
    feature, host = await _feature(tmp_path)
    await feature._persist_config()
    host.nodes[feature._config_node_id()].properties["config"]["gate"] = {"overfit_ratio": 0.5}

    restarted = ParametricSelfFeature(agent=feature.agent)
    await restarted.initialize()
    await restarted._restore_persisted_config()
    status = await restarted.parametric_self_status()

    assert restarted._gate.overfit_ratio == 1.2
    assert "overfit_ratio" in status.data["fidelity_gate"]["config_rejected"]
    assert "persisted gate config refused" in status.confirmation


async def test_a_pointer_restored_from_before_checkpoints_names_the_final_weights(tmp_path):
    feature, host = await _feature(tmp_path)
    feature._active_adapter_path = "/data/parametric_self/candidates/legacy"
    await feature._persist_config()
    del host.nodes[feature._config_node_id()].properties["config"]["active_checkpoint_id"]

    restarted = ParametricSelfFeature(agent=feature.agent)
    await restarted.initialize()
    await restarted._restore_persisted_config()
    assert restarted._active_checkpoint_id == "adapters.safetensors"


async def test_without_storage_a_gate_record_cannot_be_persisted(tmp_path):
    feature, _host = await _feature(tmp_path)
    feature.agent.storage = None
    assert await feature._persist_gate_record("run", {"decision": "promote"}) is False


async def test_malformed_history_is_never_overwritten_by_a_gate_record(tmp_path):
    feature, host = await _feature(tmp_path)
    await feature._append_run_history({"run_id": "run", "state": "in_progress"})
    node = host.nodes[feature._history_node_id()]
    node.properties["runs"] = {"not": "a list"}

    assert await feature._persist_gate_record("run", {"decision": "promote"}) is False
    assert host.nodes[feature._history_node_id()].properties["runs"] == {"not": "a list"}


# ----------------------------------------------------------------------
# History reads that feed a write must be intact
# ----------------------------------------------------------------------

async def test_a_failed_read_at_the_terminal_update_never_erases_the_gate_record(tmp_path):
    feature, host = await _feature(tmp_path)
    persist = feature._persist_gate_record

    async def _then_history_reads_fail(*args, **kwargs):
        persisted = await persist(*args, **kwargs)
        host.fail_history_reads = True
        return persisted

    feature._persist_gate_record = _then_history_reads_fail
    outcome = await feature._run_training_cycle_locked(trigger="nightly")
    host.fail_history_reads = False

    assert outcome["promoted"] is True and outcome["gate_record_persisted"] is True
    # The terminal update was skipped rather than written over unread history.
    entry = await _last_run(feature)
    assert entry["gate"] == outcome["gate"]
    assert entry["state"] == "in_progress"


async def test_run_creation_never_overwrites_malformed_history(tmp_path):
    feature, host = await _feature(tmp_path)
    await feature._append_run_history({"run_id": "old", "state": "completed"})
    host.nodes[feature._history_node_id()].properties["runs"] = {"not": "a list"}

    outcome = await feature._run_training_cycle_locked(trigger="nightly")

    assert host.nodes[feature._history_node_id()].properties["runs"] == {"not": "a list"}
    assert outcome["gate_record_persisted"] is False
    assert outcome["promoted"] is False and feature._active_adapter_path is None


@pytest.mark.parametrize("runs", [{"not": "a list"}, [{"run_id": "old"}, "not a record"], "not json"])
async def test_best_effort_history_writes_skip_rather_than_replace_unreadable_history(tmp_path, runs):
    feature, host = await _feature(tmp_path)
    await feature._append_run_history({"run_id": "old", "state": "completed"})
    host.nodes[feature._history_node_id()].properties["runs"] = runs

    await feature._append_run_history({"run_id": "new"})
    await feature._update_run_history("old", {"state": "failed"})
    await feature._reconcile_stale_runs()

    assert host.nodes[feature._history_node_id()].properties["runs"] == runs


async def test_best_effort_history_writes_skip_when_the_read_fails(tmp_path):
    feature, host = await _feature(tmp_path)
    await feature._append_run_history({"run_id": "old", "state": "in_progress"})
    before = copy.deepcopy(host.nodes[feature._history_node_id()].properties["runs"])
    host.fail_history_reads = True

    await feature._append_run_history({"run_id": "new"})
    await feature._update_run_history("old", {"state": "failed"})
    await feature._reconcile_stale_runs()

    host.fail_history_reads = False
    assert host.nodes[feature._history_node_id()].properties["runs"] == before
