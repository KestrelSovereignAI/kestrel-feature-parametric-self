"""Tests for run_nightly_cycle using a fake trainer (no MLX, Linux-friendly)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign.features.training.types import TrainingState, TrainingStatus

from kestrel_feature_parametric_self import FidelityGate, TextLoRAConfig, run_nightly_cycle as _run_nightly_cycle


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
    """Every direct cycle test still supplies the public host snapshot."""
    kwargs.setdefault("governed_snapshot", SNAPSHOT)
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


async def test_cycle_cleans_up_legacy_shared_corpus_plaintext(tmp_path):
    """#2112/P8: a host upgraded from pre-0.3.1 still has plaintext
    train.jsonl/valid.jsonl at the OLD shared work/corpus/ location (not a
    per-run subdir). A cycle must best-effort remove them so no user-derived
    plaintext lingers (F377) — while leaving per-run subdirs untouched."""
    from pathlib import Path

    work = tmp_path / "work"
    legacy = work / "corpus"
    legacy.mkdir(parents=True)
    (legacy / "train.jsonl").write_text('{"text": "old plaintext reflection"}\n')
    (legacy / "valid.jsonl").write_text('{"text": "old plaintext"}\n')
    # A per-run subdir that must NOT be touched.
    (legacy / "some-prior-run").mkdir()
    (legacy / "some-prior-run" / "train.jsonl").write_text("{}\n")

    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work),
        adapter=_FakeAdapter(log="Iter 100: Val loss 1.2"), gate=FidelityGate(max_val_loss=3.0),
        config=TextLoRAConfig(), poll_interval=0,
    )

    # Legacy flat plaintext gone.
    assert not (legacy / "train.jsonl").exists()
    assert not (legacy / "valid.jsonl").exists()
    # Per-run subdir untouched (only the pre-0.3.1 flat files are cleaned).
    assert (legacy / "some-prior-run" / "train.jsonl").exists()


async def test_legacy_corpus_cleaned_even_when_trainer_unavailable(tmp_path):
    """The cleanup must run BEFORE the trainer-availability early return — a host
    without the trainer (common) would otherwise keep the plaintext forever."""
    work = tmp_path / "work"
    legacy = work / "corpus"
    legacy.mkdir(parents=True)
    (legacy / "train.jsonl").write_text('{"text": "old plaintext"}\n')
    (legacy / "valid.jsonl").write_text('{"text": "old plaintext"}\n')

    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work),
        adapter=_FakeAdapter(available=False), gate=FidelityGate(),
        config=TextLoRAConfig(), poll_interval=0,
    )
    assert result.trained is False and "unavailable" in result.reason
    assert not (legacy / "train.jsonl").exists()
    assert not (legacy / "valid.jsonl").exists()


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


async def test_cycle_uses_supplied_adapter_id(tmp_path):
    """A caller-supplied adapter_id names the staging dir so it is introspectable."""
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    work = str(tmp_path / "work")
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=work,
        adapter=_FakeAdapter(log="Iter 100: Val loss 1.2"), gate=FidelityGate(max_val_loss=3.0),
        config=TextLoRAConfig(), poll_interval=0, adapter_id="deadbeef0001",
    )
    assert result.promoted_adapter_path == str(tmp_path / "work" / "candidates" / "deadbeef0001")


async def test_cycle_reports_training_failure(tmp_path):
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
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
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
    work = tmp_path / "work"
    adapter = _CorpusSpyAdapter(log="Iter 100: Val loss 1.2")
    result = await run_nightly_cycle(
        agent_id="emma", db_path=db, work_dir=str(work),
        adapter=adapter, gate=FidelityGate(max_val_loss=3.0),
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
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
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
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
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

        async def cancel_all(self):
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


async def test_cycle_cancellation_keeps_corpus_without_stop_confirmation(tmp_path):
    """A cancellation without a child-stop capability retains the live input."""
    db = _db_with(tmp_path, [("1", "failure", "Verbosity", "Be shorter.", "")])
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
