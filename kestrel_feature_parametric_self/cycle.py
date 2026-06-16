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
from typing import Optional, Protocol

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


async def run_nightly_cycle(
    *,
    agent_id: str,
    db_path: str,
    work_dir: str,
    adapter: _TrainerProtocol,
    gate: FidelityGate,
    config: TextLoRAConfig,
    prior_val_loss: Optional[float] = None,
    poll_interval: float = 2.0,
    max_polls: int = 5400,  # ~3h at 2s; a backstop, not a deadline
) -> CycleResult:
    """Run one corpus->train->gate cycle. Promotes nothing the gate rejects."""
    if not adapter.is_available():
        return CycleResult(False, reason="trainer unavailable on this host")

    work = Path(work_dir)
    corpus_dir = str(work / "corpus")
    # Each run trains into a UNIQUE staging dir so a rejected candidate can
    # never overwrite the currently-served adapter — the served adapter is the
    # promoted staging dir of a *prior* run, which this run never touches.
    # (Accumulating staging dirs is the adapter-lifecycle concern tracked for
    # P5 in epic #1.)
    adapter_dir = str(work / "candidates" / uuid.uuid4().hex[:12])

    stats = build_corpus(db_path, corpus_dir)
    if stats.train == 0:
        return CycleResult(
            False, reason="empty corpus — no grounded reflections to train on",
            corpus_train=0, corpus_valid=stats.valid,
        )

    config.data_dir = corpus_dir
    config.adapter_path = adapter_dir

    status = await adapter.start_training(agent_id, config)
    polls = 0
    while not status.state.is_terminal() and polls < max_polls:
        await asyncio.sleep(poll_interval)
        status = await adapter.get_status(status.job_id)
        polls += 1

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
    )
