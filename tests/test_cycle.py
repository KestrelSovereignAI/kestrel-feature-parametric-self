"""Tests for run_nightly_cycle using a fake trainer (no MLX, Linux-friendly)."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign.features.training.types import TrainingState, TrainingStatus

from kestrel_feature_parametric_self import (
    FidelityGate,
    GateBasis,
    GateOutcome,
    GateReason,
    ServedAdapter,
    TextLoRAConfig,
    run_nightly_cycle as _run_nightly_cycle,
)

ITER_200 = "0000200_adapters.safetensors"
ITER_400 = "0000400_adapters.safetensors"
FINAL = "adapters.safetensors"
RUN_FILES = (ITER_200, ITER_400, FINAL)
# What ``mlx_lm.lora --test`` scores each set of weights at on the run's
# validation set: the candidate's files by name, the untuned base, the served.
IMPROVING = {"base": 9.0, ITER_200: 1.5, ITER_400: 1.4}
OVERFIT = {"base": 9.0, ITER_200: 1.5, ITER_400: 9.9}
# Emma's June run (#11): base 8.977, iter 200 1.866, iter 400 2.688.
JUNE = {"base": 8.977, ITER_200: 1.866, ITER_400: 2.688}
# Seventy grounded insights plus the governed example give a four-row
# validation split: one scoring batch at the default batch size.
ROWS = [(str(i), "failure", f"Lesson {i}", f"Do thing {i}.", "") for i in range(70)]


def _mlx_refusal(dataset: Path, batch_size: int):
    """mlx_lm's refusal of a dataset smaller than one batch, or None."""
    rows = sum(1 for line in dataset.read_text().splitlines() if line.strip())
    if rows < batch_size:
        return f"Dataset must have at least batch_size={batch_size} examples but only has {rows}."
    return None


def _snapshot():
    assertion = SimpleNamespace(
        assertion_id="assertion:cycle", revision_id="revision:cycle",
        subject=SimpleNamespace(value="https://example.test/agent"),
        predicate=SimpleNamespace(value="https://example.test/lesson"),
        object=SimpleNamespace(lexical_form="governed lesson"),
    )
    return SimpleNamespace(
        verified=True, examples=(SimpleNamespace(
            assertion=assertion, content_hash="sha256:cycle", source_occurrences=(),
            decision=SimpleNamespace(included=True, reason=SimpleNamespace(value="included")),
        ),), snapshot_hash="sha256:snapshot", policy=SimpleNamespace(digest="sha256:policy"),
        tenant_id="tenant:test",
        checkpoint=SimpleNamespace(tenant_id="tenant:test", generation=1, latest_event_id="event:1"),
        capability_versions={"semantic_maintenance": "1"},
    )


SNAPSHOT = _snapshot()


async def run_nightly_cycle(**kwargs):
    """Every direct cycle test still supplies the public host snapshot.

    Unless a test says otherwise, nothing is served and the base model and
    served adapter stay as they were when the run started.
    """
    kwargs.setdefault("governed_snapshot", SNAPSHOT)
    kwargs.setdefault("served", None)
    basis = GateBasis(kwargs["config"].base_model, kwargs["served"])
    kwargs.setdefault("basis_now", lambda: basis)
    return await _run_nightly_cycle(**kwargs)


def _db_with(tmp_path, rows, fact=True) -> str:
    """Build a fixture cognition DB with given (type, title, desc) insight rows."""
    db = str(tmp_path / "cog.db")
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE reflection_insights (id TEXT, type TEXT, title TEXT NOT NULL, "
        "description TEXT, suggested_action TEXT)"
    )
    con.executemany(
        "INSERT INTO reflection_insights (id,type,title,description,suggested_action) VALUES (?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    return db


class _FakeAdapter:
    """A trainer that writes the run's checkpoint files, and a scorer.

    Scoring jobs score the weights file the view links to, so a loss can only
    come from the file it names. ``failing`` names the weights whose scoring
    job misbehaves as ``eval_error``/``eval_state``/``eval_log`` say; with
    None, every scoring job does.
    """

    def __init__(
        self, *, available=True, state=TrainingState.COMPLETED, scores=IMPROVING,
        served_score=5.0, files=RUN_FILES, log="", failing=None, eval_error=None,
        eval_state=TrainingState.COMPLETED, eval_log=None,
    ):
        self._available, self._state, self._files, self._log = available, state, files, log
        self._scores, self._served_score = scores, served_score
        self._failing, self._eval_error = failing, eval_error
        self._eval_state, self._eval_log = eval_state, eval_log
        self._candidate = None
        self._logs = {}
        self.trained_val_bytes = None
        self.evaluations = []

    def is_available(self):
        return self._available

    async def start_training(self, agent_id, config):
        self.trained_val_bytes = (Path(config.data_dir) / "valid.jsonl").read_bytes()
        refusal = _mlx_refusal(Path(config.data_dir) / "valid.jsonl", config.batch_size)
        if refusal is not None:
            return TrainingStatus(job_id="train", state=TrainingState.FAILED, progress=0.0, error=refusal)
        candidate = Path(config.adapter_path)
        candidate.mkdir(parents=True, exist_ok=True)
        self._candidate = candidate.resolve()
        for name in self._files:
            (candidate / name).write_bytes(b"weights")
        (candidate / "adapter_config.json").write_text("{}")
        return TrainingStatus(job_id="train", state=self._state, progress=1.0)

    def _misbehaves(self, key):
        return self._failing is None or self._failing == key

    async def start_evaluation(self, agent_id, config, *, log_dir):
        if config.adapter_path is None:
            key, weights, adapter_config = "base", None, None
        else:
            view = Path(config.adapter_path)
            weights = (view / FINAL).resolve()
            adapter_config = (view / "adapter_config.json").resolve()
            key = weights.name if weights.parent == self._candidate else "served"
        if self._eval_error is not None and self._misbehaves(key):
            raise self._eval_error
        refusal = _mlx_refusal(Path(config.data_dir) / "test.jsonl", config.batch_size)
        if refusal is not None:
            return TrainingStatus(job_id="eval-refused", state=TrainingState.FAILED, progress=0.0, error=refusal)
        self.evaluations.append({
            "key": key, "weights": weights, "adapter_config": adapter_config, "log_dir": log_dir,
            "test_bytes": (Path(config.data_dir) / "test.jsonl").read_bytes(),
        })
        job_id = f"eval-{len(self.evaluations)}"
        loss = self._served_score if key == "served" else self._scores[key]
        log = f"Testing\nTest loss {loss:.3f}, Test ppl 1.0.\n"
        state = TrainingState.COMPLETED
        if self._misbehaves(key):
            log = self._eval_log if self._eval_log is not None else log
            state = self._eval_state
        self._logs[job_id] = (log, state)
        return TrainingStatus(job_id=job_id, state=state, progress=1.0)

    async def get_status(self, job_id):
        state = self._logs[job_id][1] if job_id in self._logs else self._state
        return TrainingStatus(job_id=job_id, state=state, progress=1.0)

    def read_training_log(self, job_id):
        return self._logs[job_id][0] if job_id in self._logs else self._log

    @property
    def scored(self):
        return [evaluation["key"] for evaluation in self.evaluations]


def _served(tmp_path, checkpoint=ITER_200) -> ServedAdapter:
    served = tmp_path / "work" / "candidates" / "servedrun"
    served.mkdir(parents=True)
    (served / checkpoint).write_bytes(b"served weights")
    (served / "adapter_config.json").write_text("{}")
    return ServedAdapter(str(served), checkpoint)


async def test_cycle_promotes_when_gate_passes(tmp_path):
    db = _db_with(tmp_path, ROWS)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is True
    assert result.promoted is True
    assert result.val_loss == 1.4  # the best checkpoint's loss
    assert result.promoted_checkpoint_id == ITER_400
    assert result.promoted_adapter_path
    assert result.gate.baseline.source == "base" and result.gate.baseline.val_loss == 9.0
    assert result.gate.to_record()["base_model"] == TextLoRAConfig().base_model


async def test_every_saved_checkpoint_and_the_base_are_scored_on_the_runs_val_set(tmp_path):
    """One validation set and one scorer for every number the gate compares (R5)."""
    db = _db_with(tmp_path, ROWS)
    adapter = _FakeAdapter()
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
        adapter_id="scored",
    )
    # The final adapters.safetensors repeats the iter-400 save; it is not scored twice.
    assert adapter.scored == [ITER_200, ITER_400, "base"]
    assert all(e["test_bytes"] == adapter.trained_val_bytes for e in adapter.evaluations)
    candidate = (tmp_path / "work" / "candidates" / "scored").resolve()
    assert adapter.evaluations[0]["weights"] == candidate / ITER_200
    assert adapter.evaluations[0]["adapter_config"] == candidate / "adapter_config.json"
    assert adapter.evaluations[-1]["weights"] is None  # the untuned base: no adapter
    scoring = tmp_path / "work" / "corpus" / "scored" / "scoring"
    assert all(Path(e["log_dir"]).parent == scoring for e in adapter.evaluations)
    assert not scoring.exists()
    assert result.gate.to_record()["checkpoints"] == [
        {"checkpoint_id": ITER_200, "iteration": 200, "val_loss": 1.5, "error": None},
        {"checkpoint_id": ITER_400, "iteration": 400, "val_loss": 1.4, "error": None},
    ]


async def test_logged_losses_never_stand_in_for_the_saved_weights(tmp_path):
    """mlx_lm logs Iter 400 one optimizer step before it saves 0000400.

    Here the log reads 1.0 at iter 400 while the saved file scores 10.0. Pairing
    the log line with the file would promote unevaluated weights.
    """
    db = _db_with(tmp_path, ROWS)
    log = "Iter 1: Val loss 9.000, x\nIter 200: Val loss 1.500, x\nIter 400: Val loss 1.000, x\n"
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(log=log, scores={"base": 9.0, ITER_200: 1.5, ITER_400: 10.0}),
        gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.gate.final.checkpoint_id == ITER_400
    assert result.gate.final.val_loss == 10.0
    assert result.gate.best.checkpoint_id == ITER_200
    assert result.gate.reason is GateReason.OVERFIT
    assert result.promoted is False and result.promoted_checkpoint_id is None


async def test_the_final_weights_are_scored_when_no_periodic_save_holds_them(tmp_path):
    db = _db_with(tmp_path, ROWS)
    adapter = _FakeAdapter(files=(ITER_200, FINAL), scores={"base": 9.0, ITER_200: 1.5, FINAL: 1.45})
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=TextLoRAConfig(iters=300), poll_interval=0,
    )
    assert adapter.scored == [ITER_200, FINAL, "base"]
    assert (result.gate.final.checkpoint_id, result.gate.final.iteration) == (FINAL, 300)
    assert result.promoted is True and result.promoted_checkpoint_id == FINAL


async def test_cycle_trains_but_rejects_bad_adapter(tmp_path):
    db = _db_with(tmp_path, ROWS)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(scores=OVERFIT), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is True
    assert result.promoted is False  # 9.9 / 1.5 overfit
    assert result.gate.reason is GateReason.OVERFIT
    assert result.promoted_adapter_path is None
    assert result.promoted_checkpoint_id is None


async def test_june_run_through_the_cycle_never_promotes_its_final_checkpoint(tmp_path):
    db = _db_with(tmp_path, ROWS)
    rejected = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(scores=JUNE), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0, adapter_id="june-default",
    )
    assert rejected.gate.outcome is GateOutcome.REJECT
    assert rejected.gate.reason is GateReason.OVERFIT
    assert rejected.promoted is False and rejected.promoted_checkpoint_id is None

    relaxed = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(scores=JUNE), gate=FidelityGate(overfit_ratio=1.50),
        config=TextLoRAConfig(), poll_interval=0, adapter_id="june-relaxed",
    )
    assert relaxed.promoted is True
    assert relaxed.promoted_checkpoint_id == ITER_200
    assert relaxed.val_loss == 1.866


async def test_served_adapter_is_the_baseline_scored_on_this_runs_val_set(tmp_path):
    """R5: one validation set, scored in this run, for candidate and baseline."""
    db = _db_with(tmp_path, ROWS)
    served = _served(tmp_path)
    adapter = _FakeAdapter(served_score=1.6)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
        served=served, adapter_id="newrun",
    )
    record = result.gate.to_record()
    digest = "sha256:" + hashlib.sha256(adapter.trained_val_bytes).hexdigest()
    assert record["val_set_digest"] == digest
    assert adapter.scored == [ITER_200, ITER_400, "served"]
    assert all(e["test_bytes"] == adapter.trained_val_bytes for e in adapter.evaluations)
    assert adapter.evaluations[-1]["weights"] == (Path(served.path) / ITER_200).resolve()
    assert record["baseline_source"] == f"served:servedrun/{ITER_200}"
    assert record["baseline_val_loss"] == 1.6
    # 1.4 < 1.6 * 0.98: the candidate beats the served adapter.
    assert result.promoted is True and result.promoted_checkpoint_id == ITER_400
    # Scoring scratch is gone; the served adapter's weights are untouched.
    corpus = tmp_path / "work" / "corpus" / "newrun"
    assert not (corpus / "scoring").exists()
    assert not (corpus / "valid.jsonl").exists()
    assert (Path(served.path) / ITER_200).read_bytes() == b"served weights"


async def test_candidate_that_does_not_beat_the_served_adapter_is_rejected_on_margin(tmp_path):
    db = _db_with(tmp_path, ROWS)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(served_score=1.42), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0, served=_served(tmp_path),
    )
    assert result.trained is True
    assert result.gate.reason is GateReason.MARGIN
    assert result.promoted is False


SCORING_FAILURES = [
    ({"eval_error": RuntimeError("metal device lost")}, "metal device lost"),
    ({"eval_state": TrainingState.FAILED}, "did not complete"),
    ({"eval_log": "Testing\n"}, "no test loss"),
]


@pytest.mark.parametrize("served", [True, False], ids=["served", "base"])
@pytest.mark.parametrize("adapter_kwargs, detail", SCORING_FAILURES)
async def test_baseline_scoring_failure_is_inconclusive(tmp_path, adapter_kwargs, detail, served):
    db = _db_with(tmp_path, ROWS)
    failing = "served" if served else "base"
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(failing=failing, **adapter_kwargs), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0, adapter_id="evalfail",
        served=_served(tmp_path) if served else None,
    )
    assert result.gate.outcome is GateOutcome.INCONCLUSIVE
    assert result.gate.reason is GateReason.BASELINE_EVAL_FAILED
    assert detail in result.gate.detail
    assert result.promoted is False
    assert not (tmp_path / "work" / "corpus" / "evalfail" / "valid.jsonl").exists()


@pytest.mark.parametrize("adapter_kwargs, detail", SCORING_FAILURES)
async def test_checkpoint_scoring_failure_is_inconclusive_and_stops_scoring(tmp_path, adapter_kwargs, detail):
    db = _db_with(tmp_path, ROWS)
    adapter = _FakeAdapter(failing=ITER_200, **adapter_kwargs)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.gate.outcome is GateOutcome.INCONCLUSIVE
    assert result.gate.reason is GateReason.CANDIDATE_EVAL_FAILED
    assert detail in result.gate.detail
    assert result.promoted is False
    # Nothing after the failure is scored: the run cannot be promoted anyway.
    assert adapter.scored == ([] if "eval_error" in adapter_kwargs else [ITER_200])
    assert [c["val_loss"] for c in result.gate.to_record()["checkpoints"]] == [None, None]


async def test_a_run_that_saved_no_weights_is_inconclusive_and_scores_nothing(tmp_path):
    db = _db_with(tmp_path, ROWS)
    adapter = _FakeAdapter(files=())
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.gate.reason is GateReason.NO_CHECKPOINTS
    assert adapter.scored == []


async def test_served_checkpoint_missing_on_disk_is_inconclusive(tmp_path):
    db = _db_with(tmp_path, ROWS)
    served = _served(tmp_path)
    (Path(served.path) / ITER_200).unlink()
    adapter = _FakeAdapter()
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
        served=served,
    )
    assert result.gate.outcome is GateOutcome.INCONCLUSIVE
    assert result.gate.reason is GateReason.BASELINE_EVAL_FAILED
    assert "not on disk" in result.gate.detail
    assert "served" not in adapter.scored


async def test_a_served_adapter_quarantined_during_training_is_never_loaded(tmp_path):
    """Quarantine clears the served pointer; the old baseline is refused, not scored."""
    db = _db_with(tmp_path, ROWS)
    served = _served(tmp_path)
    adapter = _FakeAdapter(served_score=50.0)
    config = TextLoRAConfig()
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=config, poll_interval=0, served=served,
        basis_now=lambda: GateBasis(config.base_model, served if adapter.trained_val_bytes is None else None),
    )
    assert "served" not in adapter.scored
    assert result.gate.outcome is GateOutcome.INCONCLUSIVE
    assert result.gate.reason is GateReason.BASELINE_CHANGED
    record = result.gate.to_record()
    assert record["baseline_source"] == f"served:servedrun/{ITER_200}"
    assert record["baseline_val_loss"] is None
    assert "to nothing served" in record["baseline_changed"]
    assert result.promoted is False and result.promoted_checkpoint_id is None


async def test_a_baseline_that_changes_while_it_is_scored_is_not_compared(tmp_path):
    db = _db_with(tmp_path, ROWS)
    served = _served(tmp_path)
    adapter = _FakeAdapter(served_score=50.0)
    config = TextLoRAConfig()
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=config, poll_interval=0, served=served,
        basis_now=lambda: GateBasis(config.base_model, None if "served" in adapter.scored else served),
    )
    assert adapter.scored[-1] == "served"
    assert result.gate.reason is GateReason.BASELINE_CHANGED
    assert result.gate.to_record()["baseline_val_loss"] == 50.0  # scored, then refused
    assert result.promoted is False


async def test_a_base_model_reconfigured_during_the_run_is_inconclusive(tmp_path):
    db = _db_with(tmp_path, ROWS)
    adapter = _FakeAdapter()
    config = TextLoRAConfig()
    trained_on = config.base_model
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=config, poll_interval=0,
        basis_now=lambda: GateBasis(trained_on if adapter.trained_val_bytes is None else "other/base", None),
    )
    assert "base" not in adapter.scored
    assert result.gate.reason is GateReason.BASELINE_CHANGED
    assert "base model changed" in result.gate.detail
    assert result.gate.to_record()["base_model"] == trained_on
    assert result.promoted is False


async def test_empty_validation_split_is_inconclusive_and_trains_nothing(tmp_path):
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    adapter = _FakeAdapter()
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.corpus_valid == 0
    assert result.trained is False
    assert adapter.trained_val_bytes is None
    assert adapter.scored == []
    assert result.gate.outcome is GateOutcome.INCONCLUSIVE
    assert result.gate.reason is GateReason.VAL_SET_EMPTY
    assert result.reason.startswith("fidelity gate inconclusive")


@pytest.mark.parametrize("rows", [5, 40], ids=["one-row", "three-rows"])
async def test_a_validation_split_smaller_than_one_batch_is_inconclusive_and_trains_nothing(tmp_path, rows):
    """mlx_lm cannot score fewer examples than one batch, so nothing is trained or scored."""
    db = _db_with(tmp_path, ROWS[:rows])
    adapter = _FakeAdapter()
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=adapter, gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
    )
    assert 0 < result.corpus_valid < TextLoRAConfig().batch_size
    assert result.trained is False and result.promoted is False
    assert adapter.trained_val_bytes is None
    assert adapter.scored == []
    assert result.gate.outcome is GateOutcome.INCONCLUSIVE
    assert result.gate.reason is GateReason.VAL_SET_TOO_SMALL
    record = result.gate.to_record()
    assert (record["val_set_examples"], record["val_set_batch_size"]) == (result.corpus_valid, 4)


async def test_the_fake_trainer_refuses_a_set_smaller_than_one_batch_as_mlx_lm_does(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "valid.jsonl").write_text('{"text": "one"}\n' * 3)
    (corpus / "test.jsonl").write_text('{"text": "one"}\n' * 3)
    config = TextLoRAConfig(data_dir=str(corpus), adapter_path=str(tmp_path / "adapter"))
    adapter = _FakeAdapter()

    trained = await adapter.start_training("emma", config)
    scored = await adapter.start_evaluation("emma", config, log_dir=str(tmp_path / "log"))

    for status in (trained, scored):
        assert status.state is TrainingState.FAILED
        assert status.error == "Dataset must have at least batch_size=4 examples but only has 3."


async def test_cycle_noop_when_trainer_unavailable(tmp_path):
    db = _db_with(tmp_path, ROWS)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(available=False), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is False
    assert "unavailable" in result.reason


async def test_cycle_noop_on_empty_corpus(tmp_path):
    # only a non-grounded 'anomaly' insight + no fact -> grounded corpus is empty
    db = _db_with(tmp_path, [("1", "anomaly", "Musing", "A stray thought.", "")], fact=False)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        governed_snapshot=SimpleNamespace(
            verified=True, examples=(), snapshot_hash="sha256:empty",
            policy=SimpleNamespace(digest="sha256:policy"),
            tenant_id="tenant:test",
            checkpoint=SimpleNamespace(tenant_id="tenant:test", generation=1, latest_event_id="event:1"),
            capability_versions={"semantic_maintenance": "1"},
        ),
        adapter=_FakeAdapter(), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is False
    assert "empty corpus" in result.reason


async def test_cycle_retains_unknown_owner_legacy_shared_corpus_plaintext(tmp_path):
    """#2112/P8: a host upgraded from pre-0.3.1 still has plaintext
    train.jsonl/valid.jsonl at the OLD shared work/corpus/ location (not a
        per-run subdir). Without proof an old child exited, a new cycle must
        retain them for verified/operator cleanup — while leaving per-run
        subdirs untouched."""
    from pathlib import Path

    work = tmp_path / "work"
    legacy = work / "corpus"
    legacy.mkdir(parents=True)
    (legacy / "train.jsonl").write_text('{"text": "old plaintext reflection"}\n')
    (legacy / "valid.jsonl").write_text('{"text": "old plaintext"}\n')
    # A per-run subdir that must NOT be touched.
    (legacy / "some-prior-run").mkdir()
    (legacy / "some-prior-run" / "train.jsonl").write_text("{}\n")

    db = _db_with(tmp_path, ROWS)
    await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work),
        adapter=_FakeAdapter(), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )

    # Legacy flat plaintext retained: an unknown old process may still read it.
    assert (legacy / "train.jsonl").exists()
    assert (legacy / "valid.jsonl").exists()
    # Per-run subdir untouched (only the pre-0.3.1 flat files are cleaned).
    assert (legacy / "some-prior-run" / "train.jsonl").exists()


async def test_legacy_corpus_retained_even_when_trainer_unavailable(tmp_path):
    """Availability cannot prove an old trainer is not still reading legacy input."""
    work = tmp_path / "work"
    legacy = work / "corpus"
    legacy.mkdir(parents=True)
    (legacy / "train.jsonl").write_text('{"text": "old plaintext"}\n')
    (legacy / "valid.jsonl").write_text('{"text": "old plaintext"}\n')

    db = _db_with(tmp_path, ROWS)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work),
        adapter=_FakeAdapter(available=False), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is False and "unavailable" in result.reason
    assert result.legacy_corpus_retained is True
    assert (legacy / "train.jsonl").exists()
    assert (legacy / "valid.jsonl").exists()


async def test_each_run_stages_in_a_unique_dir(tmp_path):
    """A later (rejected) run must not overwrite an earlier promoted adapter."""
    db = _db_with(tmp_path, ROWS)
    work = str(tmp_path / "work")

    first = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=work,
        adapter=_FakeAdapter(), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    second = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=work,
        adapter=_FakeAdapter(scores=OVERFIT), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )

    assert first.promoted is True and first.promoted_adapter_path
    assert second.promoted is False
    # the served adapter (first) is a distinct dir the second run never wrote to
    assert second.promoted_adapter_path is None


async def test_cycle_uses_supplied_adapter_id(tmp_path):
    """A caller-supplied adapter_id names the staging dir so it is introspectable."""
    db = _db_with(tmp_path, ROWS)
    work = str(tmp_path / "work")
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=work,
        adapter=_FakeAdapter(), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0, adapter_id="deadbeef0001",
    )
    assert result.promoted_adapter_path == str(tmp_path / "work" / "candidates" / "deadbeef0001")


async def test_cycle_reports_training_failure(tmp_path):
    db = _db_with(tmp_path, ROWS)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(state=TrainingState.FAILED), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is False
    assert "did not complete" in result.reason


class _CorpusSpyAdapter(_FakeAdapter):
    """Records whether the corpus existed at training time, to prove the files
    are built + consumed BEFORE the cycle deletes them (F377)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.train_existed_during = None

    async def start_training(self, agent_id, config):
        from pathlib import Path
        self.train_existed_during = (Path(config.data_dir) / "train.jsonl").exists()
        return await super().start_training(agent_id, config)


async def test_cycle_deletes_corpus_after_training(tmp_path):
    from pathlib import Path
    db = _db_with(tmp_path, ROWS)
    work = tmp_path / "work"
    adapter = _CorpusSpyAdapter()
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work),
        adapter=adapter, gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0, adapter_id="delrun",
    )
    assert result.trained is True
    # The corpus existed during training (built + consumed)...
    assert adapter.train_existed_during is True
    # ...and is deleted afterwards — no durable plaintext user content (F377).
    corpus = work / "corpus" / "delrun"  # per-run corpus dir
    assert not (corpus / "train.jsonl").exists()
    assert not (corpus / "valid.jsonl").exists()


async def test_cycle_keeps_corpus_when_training_still_running(tmp_path):
    """codex P2: if the poll budget is exhausted while training is still
    non-terminal, the corpus must NOT be deleted — the live subprocess still
    needs its input files (F377 deletion only after terminal)."""
    from pathlib import Path
    db = _db_with(tmp_path, ROWS)
    work = tmp_path / "work"
    # Adapter stuck in TRAINING (never terminal); tiny poll budget → times out.
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work),
        adapter=_FakeAdapter(state=TrainingState.TRAINING), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0, max_polls=1, adapter_id="keeprun",
    )
    assert result.trained is False
    corpus = work / "corpus" / "keeprun"  # per-run corpus dir
    assert (corpus / "train.jsonl").exists()  # preserved for the live trainer





async def test_cycle_cancellation_stops_live_child_before_deleting_corpus(tmp_path, monkeypatch):
    """Arbitrary task cancellation cannot delete input under a live trainer."""
    db = _db_with(tmp_path, ROWS)
    work = tmp_path / "work"
    events = []

    class _LiveChild(_FakeAdapter):
        def __init__(self):
            super().__init__(state=TrainingState.TRAINING)
            self.started = asyncio.Event()
            self.corpus = None
            self.stopped = False

        async def start_training(self, agent_id, config):
            self.corpus = Path(config.data_dir)
            self.started.set()
            return await super().start_training(agent_id, config)

        async def get_status(self, job_id):
            await asyncio.Event().wait()

        async def cancel(self, job_id):
            assert self.corpus is not None
            assert (self.corpus / "train.jsonl").exists()
            self.stopped = True
            events.append("child-stopped")
            return 1

    adapter = _LiveChild()
    corpus = work / "corpus" / "cancelrun"
    import kestrel_feature_parametric_self.cycle as cycle_module

    original_delete = cycle_module._delete_corpus

    def _ordered_delete(path):
        if path == str(corpus) and (corpus / "train.jsonl").exists():
            assert adapter.stopped
            events.append("corpus-deleted")
        original_delete(path)

    monkeypatch.setattr(cycle_module, "_delete_corpus", _ordered_delete)
    task = asyncio.create_task(run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work), adapter=adapter,
        gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
        max_polls=5, adapter_id="cancelrun",
    ))
    await asyncio.wait_for(adapter.started.wait(), timeout=2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["child-stopped", "corpus-deleted"]
    assert not (corpus / "train.jsonl").exists()


async def test_cycle_cancellation_keeps_corpus_when_live_child_wont_confirm_stop(tmp_path):
    """A failed child-stop confirmation retains the live input."""
    db = _db_with(tmp_path, ROWS)
    work = tmp_path / "work"

    class _UnstoppableChild(_FakeAdapter):
        def __init__(self):
            super().__init__(state=TrainingState.TRAINING)
            self.started = asyncio.Event()

        async def start_training(self, agent_id, config):
            self.started.set()
            return await super().start_training(agent_id, config)

        async def get_status(self, job_id):
            await asyncio.Event().wait()

        async def cancel(self, job_id):
            return False

    adapter = _UnstoppableChild()
    task = asyncio.create_task(run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work), adapter=adapter,
        gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0,
        max_polls=5, adapter_id="retainrun",
    ))
    await asyncio.wait_for(adapter.started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    corpus = work / "corpus" / "retainrun"
    assert (corpus / "train.jsonl").exists()


@pytest.mark.parametrize("failing", [ITER_200, "served"])
async def test_cancellation_during_scoring_stops_that_child_before_deleting_corpus(tmp_path, failing):
    """A scoring child reads this run's validation set, like the trainer did."""
    db = _db_with(tmp_path, ROWS)
    corpus = tmp_path / "work" / "corpus" / "evalcancel"
    cancelled = []

    class _LiveEvaluation(_FakeAdapter):
        def __init__(self):
            super().__init__(failing=failing, eval_state=TrainingState.TRAINING)
            self.started = asyncio.Event()

        async def start_evaluation(self, agent_id, config, *, log_dir):
            status = await super().start_evaluation(agent_id, config, log_dir=log_dir)
            if status.state is TrainingState.TRAINING:
                self.started.set()
            return status

        async def get_status(self, job_id):
            status = await super().get_status(job_id)
            if status.state is TrainingState.TRAINING:
                await asyncio.Event().wait()
            return status

        async def cancel(self, job_id):
            assert (corpus / "valid.jsonl").exists()
            cancelled.append(job_id)
            return True

    adapter = _LiveEvaluation()
    task = asyncio.create_task(run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"), adapter=adapter,
        gate=FidelityGate(), config=TextLoRAConfig(), poll_interval=0, max_polls=5,
        served=_served(tmp_path), adapter_id="evalcancel",
    ))
    await asyncio.wait_for(adapter.started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == [f"eval-{len(adapter.evaluations)}"]
    assert adapter.scored[-1] == failing
    assert not (corpus / "valid.jsonl").exists()
    assert not (corpus / "scoring").exists()


@pytest.mark.parametrize("failing", [ITER_200, "served"])
async def test_scoring_poll_timeout_retains_the_corpus(tmp_path, failing):
    db = _db_with(tmp_path, ROWS)
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(failing=failing, eval_state=TrainingState.TRAINING), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0, max_polls=1,
        served=_served(tmp_path), adapter_id="evalslow",
    )
    assert result.training_active is True
    assert result.promoted is False
    assert "still active" in result.reason
    assert (tmp_path / "work" / "corpus" / "evalslow" / "valid.jsonl").exists()
