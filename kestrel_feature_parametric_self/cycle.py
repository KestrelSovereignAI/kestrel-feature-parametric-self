"""Nightly training cycle: corpus -> train -> score -> fidelity gate -> promote.

Orchestrates one nightly run end to end. Dependencies (the training adapter,
the fidelity gate) are injected so the whole cycle is unit-testable with fakes,
without MLX or a real training run. The feature wires the real
``LocalMLXAdapter`` + ``FidelityGate``.

This is the body of the sleep-cycle hook (§6 of the design doc): it runs after
consolidation, trains a candidate adapter on the night's reflections, and
reports whether the gate (§5.2) promotes the run's best checkpoint. The feature
records that decision before it applies any promotion.

mlx_lm logs its ``Iter N`` validation loss before optimizer step N and saves
checkpoint N after it, so the training log never scores a saved file. Once
training ends, the cycle scores every saved checkpoint, the final weights and
the baseline with ``mlx_lm.lora --test`` on the run's validation set, while that
set still exists. Each loss the gate compares is a score of the weights it names.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Tuple

from kestrel_sovereign.features.training.types import TrainingState, TrainingStatus

from .corpus import build_corpus
from .fidelity import (
    BASE_SOURCE,
    FINAL_ADAPTER_FILE,
    Baseline,
    Checkpoint,
    FidelityGate,
    GateDecision,
    ValSet,
    parse_test_loss,
    saved_checkpoints,
    served_source,
)
from .text_types import TextLoRAConfig

ADAPTER_CONFIG_FILE = "adapter_config.json"
# Per-run scoring scratch: a ``test.jsonl`` link to this run's ``valid.jsonl``,
# and for each scored set of weights a view that exposes one weights file as
# the ``adapters.safetensors`` mlx_lm loads, next to that job's log. Links and
# logs only; no plaintext copy.
_SCORING_DIR = "scoring"


class TrainingShutdownIncomplete(asyncio.CancelledError):
    """Cancellation where the trainer child could not be confirmed stopped."""

    def __init__(self, reason: str, *, corpus_path: str | None = None) -> None:
        super().__init__(reason)
        self.corpus_path = corpus_path


class TrainingStillActive(RuntimeError):
    """Polling ended while a trainer child may still be reading its corpus."""

    def __init__(self, reason: str, *, corpus_path: str) -> None:
        super().__init__(reason)
        self.corpus_path = corpus_path


class _TrainerProtocol(Protocol):
    """The slice of LocalMLXAdapter the cycle needs (so fakes can stand in)."""

    def is_available(self) -> bool: ...
    async def start_training(self, agent_id: str, config: TextLoRAConfig) -> TrainingStatus: ...
    async def start_evaluation(
        self, agent_id: str, config: TextLoRAConfig, *, log_dir: str,
    ) -> TrainingStatus: ...
    async def get_status(self, job_id: str) -> TrainingStatus: ...
    def read_training_log(self, job_id: str) -> str: ...
    async def cancel(self, job_id: str) -> bool: ...


class _ScoringStillActive(Exception):
    """A scoring child outlived the poll backstop."""


class _ChildJobs:
    """The cycle's trainer and scoring children, run one at a time.

    ``active`` is True from a child's start until it is seen terminal or its
    stop is confirmed. While it is, that child may still be reading the corpus.
    """

    def __init__(self, adapter: _TrainerProtocol, poll_interval: float, max_polls: int) -> None:
        self._adapter = adapter
        self._poll_interval = poll_interval
        self._max_polls = max_polls
        self._job_id: Optional[str] = None
        self.active = False

    async def wait(self, status: TrainingStatus) -> TrainingStatus:
        """Track a just-started child until it is terminal or the backstop runs out."""
        self._job_id = status.job_id
        self.active = True
        status = await _await_terminal(self._adapter, status, self._poll_interval, self._max_polls)
        self.active = not status.state.is_terminal()
        return status

    async def stop(self) -> None:
        """Stop the live child. ``active`` clears only on confirmation for it.

        ``cancel(job_id)`` returns an explicit confirmation for THIS cycle's
        child. A bulk count (including zero) cannot prove that this particular
        process stopped, so it is deliberately not accepted as deletion proof.
        """
        cancel = getattr(self._adapter, "cancel", None)
        if not self.active or not callable(cancel):
            return
        try:
            stopped = await cancel(self._job_id)
        except Exception:
            # F377 safety boundary: a failed termination attempt means the
            # corpus remains available to a potentially live child.
            return
        self.active = not bool(stopped)


@dataclass(frozen=True)
class ServedAdapter:
    """The adapter currently served: its candidate dir and promoted checkpoint."""

    path: str
    checkpoint_id: str

    @property
    def source(self) -> str:
        return served_source(Path(self.path).name, self.checkpoint_id)


@dataclass
class CycleResult:
    """Outcome of one nightly cycle."""

    trained: bool
    promoted: bool = False
    reason: str = ""
    val_loss: Optional[float] = None    # the run's best checkpoint, when there is one
    promoted_adapter_path: Optional[str] = None
    promoted_checkpoint_id: Optional[str] = None
    gate: Optional[GateDecision] = None
    corpus_train: int = 0
    corpus_valid: int = 0
    corpus_manifest_path: Optional[str] = None
    corpus_manifest_hash: Optional[str] = None
    semantic_checkpoint_generation: Optional[int] = None
    semantic_checkpoint_id: Optional[str] = None
    corpus_snapshot_hash: Optional[str] = None
    corpus_policy_digest: Optional[str] = None
    assertion_lineage: tuple[tuple[str, str], ...] = ()
    training_active: bool = False
    retained_corpus_path: Optional[str] = None
    legacy_corpus_retained: bool = False


async def run_nightly_cycle(
    *,
    agent_id: str,
    db_path: str | None,
    work_dir: str,
    governed_snapshot: Any,
    adapter: _TrainerProtocol,
    gate: FidelityGate,
    config: TextLoRAConfig,
    served: Optional[ServedAdapter] = None,
    adapter_id: Optional[str] = None,
    poll_interval: float = 2.0,
    max_polls: int = 5400,  # ~3h at 2s; a backstop, not a deadline
) -> CycleResult:
    """Run one corpus->train->score->gate cycle. Promotes nothing the gate rejects.

    ``served`` is the adapter currently served, if any. It is the gate's
    baseline; with nothing served, the baseline is the untuned base. Either is
    scored on this run's validation set exactly as the run's checkpoints are.
    """
    work = Path(work_dir)

    # A pre-0.3.1 host may have plaintext directly under ``work/corpus/``. Its
    # owner/process cannot be proven after an upgrade, so do NOT delete it here:
    # an old child could still be reading it. The feature surfaces this retained
    # cleanup requirement; only an operator/verified process lifecycle may remove
    # it safely.
    legacy_corpus_retained = any(
        (work / "corpus" / name).exists() for name in ("train.jsonl", "valid.jsonl")
    )

    if not adapter.is_available():
        return CycleResult(
            False, reason="trainer unavailable on this host",
            legacy_corpus_retained=legacy_corpus_retained,
        )
    # Each run trains into a UNIQUE staging dir so a rejected candidate can
    # never overwrite the currently-served adapter — the served adapter is the
    # promoted staging dir of a *prior* run, which this run never touches.
    # (Accumulating staging dirs is the adapter-lifecycle concern tracked for
    # P5 in epic #1.) The caller may supply the staging-dir name so it can
    # surface the in-progress run (and its candidate) before the cycle returns;
    # otherwise a fresh id is minted here.
    run_id = adapter_id or uuid.uuid4().hex[:12]
    adapter_dir = str(work / "candidates" / run_id)
    # PER-RUN corpus dir (keyed on the same run id): so a concurrent/later cycle
    # can never overwrite or delete a still-needed corpus from a prior run that
    # timed out with its trainer still alive (codex P2).
    corpus_dir = str(work / "corpus" / run_id)

    # Only safe to delete the corpus once no trainer or scoring child is (or
    # may be) still reading it.
    jobs = _ChildJobs(adapter, poll_interval, max_polls)
    try:
        stats = build_corpus(
            db_path,
            corpus_dir,
            governed_snapshot=governed_snapshot,
            manifest_dir=adapter_dir,
        )
        if stats.train == 0:
            return CycleResult(
                False, reason="empty corpus — no grounded reflections to train on",
                corpus_train=0, corpus_valid=stats.valid,
                corpus_manifest_path=stats.manifest_path,
                corpus_manifest_hash=stats.manifest_hash,
                semantic_checkpoint_generation=stats.semantic_checkpoint_generation,
                semantic_checkpoint_id=stats.semantic_checkpoint_id,
                corpus_snapshot_hash=stats.snapshot_hash,
                corpus_policy_digest=stats.policy_digest,
                assertion_lineage=stats.assertion_lineage,
            )

        val_set = read_val_set(Path(corpus_dir) / "valid.jsonl")
        if val_set is None or val_set.examples == 0:
            # Nothing could be scored: record an inconclusive gate, train nothing.
            decision = gate.evaluate(checkpoints=(), baseline=None, val_set=val_set)
            return CycleResult(
                False, reason=decision.summary, gate=decision,
                corpus_train=stats.train, corpus_valid=stats.valid,
                corpus_manifest_path=stats.manifest_path,
                corpus_manifest_hash=stats.manifest_hash,
                semantic_checkpoint_generation=stats.semantic_checkpoint_generation,
                semantic_checkpoint_id=stats.semantic_checkpoint_id,
                corpus_snapshot_hash=stats.snapshot_hash,
                corpus_policy_digest=stats.policy_digest,
                assertion_lineage=stats.assertion_lineage,
                legacy_corpus_retained=legacy_corpus_retained,
            )

        config.data_dir = corpus_dir
        config.adapter_path = adapter_dir

        status = await jobs.wait(await adapter.start_training(agent_id, config))

        # Safe to delete only once the job is terminal. If the poll backstop was
        # exhausted while still non-terminal, the trainer subprocess may still be
        # reading its input, so KEEP the corpus (codex P2) — we can't reliably
        # confirm the subprocess exited from here (a cancel() only requests
        # termination). Cleaning up a runaway job's working files is the
        # adapter's lifecycle responsibility (epic #1 follow-up).
        if jobs.active:
            return _still_active(
                "training still active after poll timeout; corpus retained",
                stats, corpus_dir, legacy_corpus_retained,
            )

        if status.state != TrainingState.COMPLETED:
            return CycleResult(
                False, reason=f"training did not complete (state={status.state.value}; {status.error or ''})".strip(),
                corpus_train=stats.train, corpus_valid=stats.valid,
                legacy_corpus_retained=legacy_corpus_retained,
            )

        scorer = _Scorer(adapter, jobs, agent_id, config, Path(corpus_dir) / _SCORING_DIR)
        try:
            checkpoints, baseline = await scorer.score_run(
                Path(adapter_dir),
                saved_checkpoints(_weight_files(adapter_dir), final_iteration=config.iters),
                served,
            )
        except _ScoringStillActive as exc:
            return _still_active(str(exc), stats, corpus_dir, legacy_corpus_retained)

        decision = gate.evaluate(checkpoints=checkpoints, baseline=baseline, val_set=val_set)
        return CycleResult(
            trained=True,
            promoted=decision.promote,
            reason=decision.summary,
            val_loss=decision.best.val_loss if decision.best is not None else None,
            promoted_adapter_path=adapter_dir if decision.promote else None,
            promoted_checkpoint_id=decision.promoted_checkpoint_id,
            gate=decision,
            corpus_train=stats.train,
            corpus_valid=stats.valid,
            corpus_manifest_path=stats.manifest_path,
            corpus_manifest_hash=stats.manifest_hash,
            semantic_checkpoint_generation=stats.semantic_checkpoint_generation,
            semantic_checkpoint_id=stats.semantic_checkpoint_id,
            corpus_snapshot_hash=stats.snapshot_hash,
            corpus_policy_digest=stats.policy_digest,
            assertion_lineage=stats.assertion_lineage,
            legacy_corpus_retained=legacy_corpus_retained,
        )
    except asyncio.CancelledError:
        # A task cancellation alone does NOT stop the child process.  Terminate
        # and wait for the trainer or scorer before allowing finally to remove
        # plaintext corpus input; if the adapter cannot provide that
        # confirmation, preserve the corpus rather than deleting files a live
        # child may still be reading.
        await jobs.stop()
        if jobs.active:
            raise TrainingShutdownIncomplete(
                "trainer stop could not be confirmed; corpus retained",
                corpus_path=corpus_dir,
            )
        raise
    finally:
        # The corpus is transient training INPUT derived from user-authored
        # reflections/facts — don't leave train.jsonl/valid.jsonl on disk as
        # durable plaintext once the trainer is done with them (F377). Skip the
        # delete only when a started job is still running WITHOUT being torn
        # down (a live poll-timeout job, codex P2), so a running subprocess
        # never loses its input mid-run.
        if not jobs.active:
            _delete_corpus(corpus_dir)


@dataclass
class _Scorer:
    """Scores sets of weights on one run's validation set with ``mlx_lm.lora --test``."""

    adapter: _TrainerProtocol
    jobs: _ChildJobs
    agent_id: str
    config: TextLoRAConfig
    scoring_dir: Path

    async def score_run(
        self, adapter_dir: Path, saved: Sequence[Checkpoint], served: Optional[ServedAdapter],
    ) -> Tuple[Tuple[Checkpoint, ...], Baseline]:
        """Score every saved checkpoint, then the baseline.

        Scoring stops at the first failure, which makes the gate inconclusive;
        whatever follows it is left unscored.
        """
        source = served.source if served is not None else BASE_SOURCE
        if not saved:
            return (), Baseline(source)
        problem = self._stage()
        if problem is not None:
            return tuple(replace(c, error=problem) for c in saved), Baseline(source, error=problem)
        failure: Optional[str] = None
        scored = []
        for checkpoint in saved:
            if failure is None:
                loss, failure = await self._score(
                    f"candidate-{checkpoint.iteration:07d}", (adapter_dir, checkpoint.checkpoint_id),
                )
                checkpoint = replace(checkpoint, val_loss=loss, error=failure)
            scored.append(checkpoint)
        if failure is not None:
            return tuple(scored), Baseline(source)
        if served is None:
            loss, error = await self._score("base", None)
        else:
            loss, error = await self._score("served", (Path(served.path), served.checkpoint_id))
        return tuple(scored), Baseline(source, loss, error)

    def _stage(self) -> Optional[str]:
        """Link this run's validation set as the ``test.jsonl`` every score reads."""
        try:
            self.scoring_dir.mkdir(parents=True)
            (self.scoring_dir / "test.jsonl").symlink_to(Path("..") / "valid.jsonl")
        except OSError as exc:
            return f"could not stage the validation set for scoring: {exc}"
        return None

    async def _score(
        self, label: str, weights: Optional[Tuple[Path, str]],
    ) -> Tuple[Optional[float], Optional[str]]:
        """Score one set of weights: an (adapter dir, weights file) pair, or None for the base.

        Returns the loss, or None and why the weights could not be scored.
        """
        view = self.scoring_dir / label
        eval_config = TextLoRAConfig.from_dict(self.config.to_dict())
        eval_config.data_dir = str(self.scoring_dir)
        eval_config.adapter_path = None
        if weights is not None:
            problem = _stage_weights(view / "adapter", *weights)
            if problem is not None:
                return None, problem
            eval_config.adapter_path = str(view / "adapter")
        try:
            started = await self.adapter.start_evaluation(self.agent_id, eval_config, log_dir=str(view))
        except Exception as exc:  # noqa: BLE001
            # Fail closed: whatever stopped the scoring job from starting is
            # recorded and makes the gate inconclusive.
            return None, f"{type(exc).__name__}: {exc}"
        status = await self.jobs.wait(started)
        if self.jobs.active:
            raise _ScoringStillActive(f"scoring {label} still active after poll timeout; corpus retained")
        if status.state != TrainingState.COMPLETED:
            return None, f"evaluation did not complete (state={status.state.value}; {status.error or ''})".strip()
        loss = parse_test_loss(self.adapter.read_training_log(status.job_id))
        if loss is None:
            return None, "evaluation reported no test loss"
        return loss, None


def read_val_set(path: Path) -> Optional[ValSet]:
    """Identify the validation file every score in this run is computed on."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return ValSet(
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
        examples=sum(1 for line in data.splitlines() if line.strip()),
    )


def _weight_files(adapter_dir: str) -> list[str]:
    directory = Path(adapter_dir)
    if not directory.is_dir():
        return []
    return [
        entry.name for entry in directory.iterdir()
        if entry.is_file() and entry.name.endswith(".safetensors")
    ]


async def _await_terminal(
    adapter: _TrainerProtocol, status: TrainingStatus, poll_interval: float, max_polls: int,
) -> TrainingStatus:
    """Poll a job until it is terminal or the poll backstop runs out."""
    polls = 0
    while not status.state.is_terminal() and polls < max_polls:
        await asyncio.sleep(poll_interval)
        status = await adapter.get_status(status.job_id)
        polls += 1
    return status


def _still_active(reason: str, stats: Any, corpus_dir: str, legacy_corpus_retained: bool) -> CycleResult:
    return CycleResult(
        False,
        reason=reason,
        corpus_train=stats.train,
        corpus_valid=stats.valid,
        training_active=True,
        retained_corpus_path=corpus_dir,
        legacy_corpus_retained=legacy_corpus_retained,
    )


def _stage_weights(adapter_view: Path, weights_dir: Path, checkpoint_id: str) -> Optional[str]:
    """Expose one weights file, with its adapter config, as a loadable adapter dir.

    Returns why the weights cannot be scored, or None once staged.
    """
    if Path(checkpoint_id).name != checkpoint_id:
        return f"checkpoint id {checkpoint_id!r} is not a file name"
    weights = weights_dir / checkpoint_id
    adapter_config = weights_dir / ADAPTER_CONFIG_FILE
    if not weights.is_file():
        return f"checkpoint {checkpoint_id} is not on disk"
    if not adapter_config.is_file():
        return f"adapter {weights_dir.name} has no {ADAPTER_CONFIG_FILE}"
    try:
        adapter_view.mkdir(parents=True)
        (adapter_view / ADAPTER_CONFIG_FILE).symlink_to(adapter_config.resolve())
        (adapter_view / FINAL_ADAPTER_FILE).symlink_to(weights.resolve())
    except OSError as exc:
        return f"could not stage checkpoint {checkpoint_id} for scoring: {exc}"
    return None


def _delete_corpus(corpus_dir: str) -> None:
    """Remove the transient corpus files (best effort; never breaks the cycle)."""
    corpus = Path(corpus_dir)
    for name in ("train.jsonl", "valid.jsonl"):
        try:
            (corpus / name).unlink(missing_ok=True)
        except OSError:
            pass
    # Links and scoring logs only; rmtree removes links, never their targets.
    try:
        shutil.rmtree(corpus / _SCORING_DIR)
    except FileNotFoundError:
        pass
    except OSError:
        pass
