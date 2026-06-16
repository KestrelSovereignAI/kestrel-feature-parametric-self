"""Tests for run_nightly_cycle using a fake trainer (no MLX, Linux-friendly)."""

from __future__ import annotations

import sqlite3

import pytest

from kestrel_sovereign.features.training.types import TrainingState, TrainingStatus

from kestrel_feature_parametric_self import FidelityGate, TextLoRAConfig, run_nightly_cycle


def _db_with(tmp_path, rows, fact=True) -> str:
    """Build a fixture cognition DB with given (type, title, desc) insight rows."""
    db = str(tmp_path / "cog.db")
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE reflection_insights (id TEXT, type TEXT, title TEXT NOT NULL, "
        "description TEXT, suggested_action TEXT)"
    )
    con.execute("CREATE TABLE graph_nodes (node_id TEXT, node_type TEXT, label TEXT, properties TEXT)")
    con.executemany(
        "INSERT INTO reflection_insights (id,type,title,description,suggested_action) VALUES (?,?,?,?,?)",
        rows,
    )
    if fact:
        con.execute(
            "INSERT INTO graph_nodes (node_id,node_type,label,properties) VALUES ('n','learned_fact','f','{}')"
        )
    con.commit()
    con.close()
    return db


class _FakeAdapter:
    def __init__(self, *, available=True, state=TrainingState.COMPLETED, log="Iter 100: Val loss 1.500"):
        self._available, self._state, self._log = available, state, log

    def is_available(self):
        return self._available

    async def start_training(self, agent_id, config):
        return TrainingStatus(job_id="job-1", state=self._state, progress=1.0)

    async def get_status(self, job_id):
        return TrainingStatus(job_id=job_id, state=self._state, progress=1.0)

    def read_training_log(self, job_id):
        return self._log


async def test_cycle_promotes_when_gate_passes(tmp_path):
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(log="Iter 100: Val loss 1.2"), gate=FidelityGate(max_val_loss=3.0),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is True
    assert result.promoted is True
    assert result.val_loss == 1.2
    assert result.promoted_adapter_path


async def test_cycle_trains_but_rejects_bad_adapter(tmp_path):
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(log="Iter 100: Val loss 9.9"), gate=FidelityGate(max_val_loss=3.0),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is True
    assert result.promoted is False  # 9.9 over ceiling
    assert result.promoted_adapter_path is None


async def test_cycle_noop_when_trainer_unavailable(tmp_path):
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
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
        adapter=_FakeAdapter(), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is False
    assert "empty corpus" in result.reason


async def test_each_run_stages_in_a_unique_dir(tmp_path):
    """A later (rejected) run must not overwrite an earlier promoted adapter."""
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    work = str(tmp_path / "work")

    first = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=work,
        adapter=_FakeAdapter(log="Iter 100: Val loss 1.2"), gate=FidelityGate(max_val_loss=3.0),
        config=TextLoRAConfig(), poll_interval=0,
    )
    second = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=work,
        adapter=_FakeAdapter(log="Iter 100: Val loss 9.9"), gate=FidelityGate(max_val_loss=3.0),
        config=TextLoRAConfig(), poll_interval=0,
    )

    assert first.promoted is True and first.promoted_adapter_path
    assert second.promoted is False
    # the served adapter (first) is a distinct dir the second run never wrote to
    assert second.promoted_adapter_path is None


async def test_cycle_reports_training_failure(tmp_path):
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(tmp_path / "work"),
        adapter=_FakeAdapter(state=TrainingState.FAILED), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is False
    assert "did not complete" in result.reason
