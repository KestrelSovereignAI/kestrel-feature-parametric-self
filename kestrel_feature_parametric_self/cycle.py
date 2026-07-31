"""Nightly training cycle: corpus -> train -> fidelity gate -> promote.

Orchestrates one nightly run end to end. Dependencies (the training adapter,
the fidelity gate) are injected so the whole cycle is unit-testable with fakes,
without MLX or a real training run. The feature wires the real
``LocalMLXAdapter`` + ``FidelityGate``.

This is the body of the sleep-cycle hook (§6 of the design doc): it runs after
consolidation, trains a candidate adapter on the night's reflections, and
promotes it only if it clears the gate (§5.2).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from kestrel_sovereign.features.training.types import TrainingState, TrainingStatus

from .corpus import build_corpus
from .fidelity import FidelityGate, parse_final_val_loss
from .text_types import TextLoRAConfig


class _TrainerProtocol(Protocol):
    """The slice of LocalMLXAdapter the cycle needs (so fakes can stand in)."""

    def is_available(self) -> bool: ...
    async def start_training(self, agent_id: str, config: TextLoRAConfig) -> TrainingStatus: ...
    async def get_status(self, job_id: str) -> TrainingStatus: ...
    def read_training_log(self, job_id: str) -> str: ...
    async def cancel_all(self) -> int: ...


@dataclass
class CycleResult:
    """Outcome of one nightly cycle."""

    trained: bool
    promoted: bool = False
    reason: str = ""
    val_loss: Optional[float] = None
    promoted_adapter_path: Optional[str] = None
    corpus_train: int = 0
    corpus_valid: int = 0
    corpus_manifest_path: Optional[str] = None
    corpus_manifest_hash: Optional[str] = None
    semantic_checkpoint_generation: Optional[int] = None
    semantic_checkpoint_id: Optional[str] = None
    corpus_snapshot_hash: Optional[str] = None
    corpus_policy_digest: Optional[str] = None
    assertion_lineage: tuple[tuple[str, str], ...] = ()


async def run_nightly_cycle(
    *,
    agent_id: str,
    db_path: str | None,
    work_dir: str,
    governed_snapshot: Any,
    adapter: _TrainerProtocol,
    gate: FidelityGate,
    config: TextLoRAConfig,
    prior_val_loss: Optional[float] = None,
    adapter_id: Optional[str] = None,
    poll_interval: float = 2.0,
    max_polls: int = 5400,  # ~3h at 2s; a backstop, not a deadline
) -> CycleResult:
    """Run one corpus->train->gate cycle. Promotes nothing the gate rejects."""
    work = Path(work_dir)

    # One-time cleanup of the PRE-0.3.1 shared corpus location (F377/P8): older
    # code wrote plaintext train.jsonl/valid.jsonl DIRECTLY at ``work/corpus/``
    # (not a per-run subdir), and a host upgraded from that version still carries
    # that plaintext user-derived corpus on disk. Do this FIRST — before the
    # trainer-availability early return — so the lingering plaintext is removed
    # even on hosts where the trainer is unavailable or broken (where the cycle
    # never trains). ``_delete_corpus`` only touches ``<dir>/train.jsonl``/
    # ``valid.jsonl``, never the new per-run ``work/corpus/<run_id>/`` subdirs.
    _delete_corpus(str(work / "corpus"))

    if not adapter.is_available():
        return CycleResult(False, reason="trainer unavailable on this host")
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

    # Only safe to delete the corpus once no training subprocess is (or may be)
    # still reading it. True while a started job hasn't reached a terminal state.
    training_active = False
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

        config.data_dir = corpus_dir
        config.adapter_path = adapter_dir

        status = await adapter.start_training(agent_id, config)
        training_active = True
        polls = 0
        while not status.state.is_terminal() and polls < max_polls:
            await asyncio.sleep(poll_interval)
            status = await adapter.get_status(status.job_id)
            polls += 1

        # Safe to delete only once the job is terminal. If the poll backstop was
        # exhausted while still non-terminal, the trainer subprocess may still be
        # reading its input, so KEEP the corpus (codex P2) — we can't reliably
        # confirm the subprocess exited from here (a cancel() only requests
        # termination). Cleaning up a runaway job's working files is the
        # adapter's lifecycle responsibility (epic #1 follow-up).
        training_active = not status.state.is_terminal()

        if status.state != TrainingState.COMPLETED:
            return CycleResult(
                False, reason=f"training did not complete (state={status.state.value}; {status.error or ''})".strip(),
                corpus_train=stats.train, corpus_valid=stats.valid,
            )

        val_loss = parse_final_val_loss(adapter.read_training_log(status.job_id))
        decision = gate.evaluate(val_loss, prior_val_loss)
        return CycleResult(
            trained=True,
            promoted=decision.promote,
            reason=decision.reason,
            val_loss=val_loss,
            promoted_adapter_path=adapter_dir if decision.promote else None,
            corpus_train=stats.train,
            corpus_valid=stats.valid,
            corpus_manifest_path=stats.manifest_path,
            corpus_manifest_hash=stats.manifest_hash,
            semantic_checkpoint_generation=stats.semantic_checkpoint_generation,
            semantic_checkpoint_id=stats.semantic_checkpoint_id,
            corpus_snapshot_hash=stats.snapshot_hash,
            corpus_policy_digest=stats.policy_digest,
            assertion_lineage=stats.assertion_lineage,
        )
    except asyncio.CancelledError:
        # A task cancellation alone does NOT stop the child process.  Terminate
        # and wait for the trainer before allowing finally to remove plaintext
        # corpus input; if the adapter cannot provide that confirmation, preserve
        # the corpus rather than deleting files a live child may still be reading.
        cancel_all = getattr(adapter, "cancel_all", None)
        if training_active and callable(cancel_all):
            try:
                await cancel_all()
            except Exception:
                # F377 safety boundary: a failed termination attempt means the
                # corpus remains available to a potentially live child.
                pass
            else:
                training_active = False
        raise
    finally:
        # The corpus is transient training INPUT derived from user-authored
        # reflections/facts — don't leave train.jsonl/valid.jsonl on disk as
        # durable plaintext once the trainer is done with them (F377). Skip the
        # delete only when a started job is still running WITHOUT being torn
        # down (a live poll-timeout job, codex P2), so a running subprocess
        # never loses its input mid-run.
        if not training_active:
            _delete_corpus(corpus_dir)


def _delete_corpus(corpus_dir: str) -> None:
    """Remove the transient corpus files (best effort; never breaks the cycle)."""
    corpus = Path(corpus_dir)
    for name in ("train.jsonl", "valid.jsonl"):
        try:
            (corpus / name).unlink(missing_ok=True)
        except OSError:
            pass
