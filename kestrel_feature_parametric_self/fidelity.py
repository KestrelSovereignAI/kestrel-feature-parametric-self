"""Fidelity gate — decides whether a freshly trained adapter may be promoted.

This is the §5.2 promotion gate from docs/TWO_BRAIN_ARCHITECTURE.md. Its metric
is held-out validation loss (what ``mlx_lm.lora`` reports on the held-out
split), so it only measures how well an adapter fits the agent's own data. It
says nothing about identity drift; the behavioural fidelity exam (epic #1) is
the real gate and is a separate check.

The gate is relative, not an absolute ceiling (#30). It picks the run's best
checkpoint by validation loss (#11) and promotes that checkpoint only if, in
this order:

  1. every input could be evaluated. Otherwise the decision is
     ``inconclusive``, which blocks promotion; it is never a pass;
  2. the run did not overfit: ``final_loss / best_loss <= overfit_ratio``;
  3. the best checkpoint beats the baseline by the margin:
     ``best_loss < baseline_loss * (1 - margin)``.

The first failing check sets the decision and its reason. The baseline is the
currently served adapter, or the untuned base model when nothing is served,
scored on the same validation set in the same gate run. A loss stored from an
earlier run is never a baseline, and neither is one that stopped being the
baseline (a different served adapter or base model) before the decision.
Every decision names the base model its scores were measured on.

Every loss the gate compares is a score of the exact weights it names. mlx_lm
measures its in-training ``Iter N`` validation loss before optimizer step N and
saves checkpoint N after that step, so no log line describes a saved file. The
cycle therefore scores each saved checkpoint file, the final weights and the
baseline with ``mlx_lm.lora --test`` on the run's validation set (``cycle.py``).

Every decision carries a complete, JSON-safe record (``GateDecision.to_record``)
so history keeps the comparison, not just one number. The parsers and the gate
are pure functions so they are unit-testable without MLX or a real run.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Matches mlx_lm.lora lines like: "Iter 400: Val loss 2.208, Val took 42.083s"
_VAL_LOSS_RE = re.compile(r"Val loss\s+([0-9]+(?:\.[0-9]+)?)")
# mlx_lm prints losses with ``:.3f``, which renders non-finite values as
# ``nan``/``inf``. The gate's parser keeps them so the gate can refuse them.
_LOSS = r"(nan|-?inf|[0-9]+(?:\.[0-9]+)?)"
# Matches the ``--test`` summary: "Test loss 2.208, Test ppl 9.098."
_TEST_LOSS_RE = re.compile(r"Test loss\s+" + _LOSS)
# Matches the iteration counter in either Train- or Val-loss lines: "Iter 400:".
_ITER_RE = re.compile(r"Iter\s+([0-9]+)")
# mlx_lm saves a periodic checkpoint as ``{iteration:07d}_adapters.safetensors``.
_CHECKPOINT_FILE_RE = re.compile(r"^([0-9]{7})_adapters\.safetensors$")

# The file mlx_lm loads from an adapter directory, and that it overwrites with
# the final weights when a run ends.
FINAL_ADAPTER_FILE = "adapters.safetensors"
BASE_SOURCE = "base"
GATE_RECORD_SCHEMA = "parametric-self-gate-v1"


def parse_final_val_loss(log_text: str) -> Optional[float]:
    """Return the last validation loss reported in an mlx_lm.lora training log.

    Returns None if the log contains no validation loss (e.g. the run never
    reached an eval step or failed early). This is a progress/display reading
    only: it was measured one optimizer step before the weights mlx_lm saved,
    so the gate never uses it.
    """
    matches = _VAL_LOSS_RE.findall(log_text or "")
    if not matches:
        return None
    return float(matches[-1])


def parse_latest_iter(log_text: str) -> Optional[int]:
    """Return the most recent training iteration reported in the log.

    Used only to surface progress for an in-flight run (``last_seen_iter``); a
    mid-run ``Val loss`` is an intermediate snapshot, never a terminal result,
    so the iter lets an operator see the run is still advancing rather than
    mistaking an intermediate loss for the final one.
    """
    matches = _ITER_RE.findall(log_text or "")
    if not matches:
        return None
    return int(matches[-1])


def parse_test_loss(log_text: str) -> Optional[float]:
    """Return the loss an ``mlx_lm.lora --test`` evaluation reported, if any."""
    matches = _TEST_LOSS_RE.findall(log_text or "")
    if not matches:
        return None
    return float(matches[-1])


@dataclass(frozen=True)
class Checkpoint:
    """One saved set of adapter weights from a training run."""

    checkpoint_id: str              # the weights file name inside the candidate dir
    iteration: int                  # optimizer steps taken before these weights were saved
    val_loss: Optional[float]       # this file's score; None when it was not scored
    error: Optional[str] = None     # why scoring this file failed


@dataclass(frozen=True)
class Baseline:
    """What a candidate must beat, scored in the same gate run."""

    source: str                     # BASE_SOURCE or served_source(...)
    val_loss: Optional[float] = None
    error: Optional[str] = None     # why scoring the baseline failed
    changed: Optional[str] = None   # why it is no longer the baseline


@dataclass(frozen=True)
class ValSet:
    """Identity of the validation set every score in one gate run used.

    mlx_lm scores only full batches of ``batch_size`` examples and refuses a
    set smaller than one batch, so the batch size is part of what was scored.
    """

    digest: str                     # "sha256:<hex>" of the validation file's bytes
    examples: int
    batch_size: int                 # examples per scoring batch


def served_source(adapter_id: str, checkpoint_id: str) -> str:
    """Baseline source label for a served adapter's promoted checkpoint."""
    return f"served:{adapter_id}/{checkpoint_id}"


def saved_checkpoints(weight_files: Iterable[str], *, final_iteration: int) -> Tuple[Checkpoint, ...]:
    """The distinct weights a completed run saved, unscored, oldest first.

    A periodic ``{iteration:07d}_adapters.safetensors`` file holds the weights
    after that many optimizer steps. ``adapters.safetensors`` holds the weights
    after the run's last step, ``final_iteration``. It is listed only when no
    periodic file already holds those weights, so no weights are scored twice.
    """
    by_iteration: Dict[int, str] = {}
    has_final_file = False
    for name in weight_files:
        match = _CHECKPOINT_FILE_RE.match(name)
        if match:
            by_iteration[int(match.group(1))] = name
        elif name == FINAL_ADAPTER_FILE:
            has_final_file = True
    if has_final_file:
        by_iteration.setdefault(final_iteration, FINAL_ADAPTER_FILE)
    return tuple(
        Checkpoint(checkpoint_id=name, iteration=iteration, val_loss=None)
        for iteration, name in sorted(by_iteration.items())
    )


def margin_threshold(baseline_loss: float, margin: float) -> float:
    """The loss a candidate must be strictly below to beat the baseline."""
    return baseline_loss * (1.0 - margin)


def loss_ratio(final_loss: float, best_loss: float) -> float:
    """How far the run's final loss rose above its best (1.0 = no rise)."""
    return final_loss / best_loss


class GateOutcome(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


# Outcomes of a gate that actually checked a run. Inconclusive is not one.
DECIDED_OUTCOMES = frozenset({GateOutcome.PROMOTE.value, GateOutcome.REJECT.value})


class GateReason(str, Enum):
    PASSED = "passed"
    OVERFIT = "overfit"
    MARGIN = "margin"
    VAL_SET_MISSING = "val_set_missing"
    VAL_SET_EMPTY = "val_set_empty"
    VAL_SET_TOO_SMALL = "val_set_too_small"
    NO_CHECKPOINTS = "no_checkpoints"
    CANDIDATE_EVAL_FAILED = "candidate_eval_failed"
    CANDIDATE_LOSS_INVALID = "candidate_loss_invalid"
    NO_CHECKPOINT_VAL_LOSS = "no_checkpoint_val_loss"
    FINAL_CHECKPOINT_NO_VAL_LOSS = "final_checkpoint_no_val_loss"
    BASELINE_UNAVAILABLE = "baseline_unavailable"
    BASELINE_CHANGED = "baseline_changed"
    BASELINE_EVAL_FAILED = "baseline_eval_failed"
    BASELINE_LOSS_INVALID = "baseline_loss_invalid"


GATE_RECORD_FIELDS = (
    "schema",
    "decision",
    "reason",
    "detail",
    "best_checkpoint_id",
    "best_iteration",
    "best_val_loss",
    "final_checkpoint_id",
    "final_iteration",
    "final_val_loss",
    "checkpoints",
    "baseline_source",
    "baseline_val_loss",
    "baseline_error",
    "baseline_changed",
    "base_model",
    "margin",
    "required_below",
    "overfit_check",
    "overfit_ratio",
    "ratio",
    "val_set_digest",
    "val_set_examples",
    "val_set_batch_size",
    "promoted_checkpoint_id",
)


def is_usable_loss(value: Any) -> bool:
    """A cross-entropy loss the gate can compare: a finite, positive number."""
    return _is_real(value) and math.isfinite(value) and value > 0.0


def _json_number(value: Optional[float]) -> Any:
    """JSON-safe number: non-finite values are kept visible as strings."""
    if value is None or math.isfinite(value):
        return value
    return str(value)


def _fmt(value: float) -> str:
    return f"{value:.3f}" if math.isfinite(value) else str(value)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class GateDecision:
    """Outcome of one fidelity evaluation, with everything it compared."""

    outcome: GateOutcome
    reason: GateReason
    detail: str
    gate: "FidelityGate"
    checkpoints: Tuple[Checkpoint, ...] = ()
    best: Optional[Checkpoint] = None
    final: Optional[Checkpoint] = None
    baseline: Optional[Baseline] = None
    ratio: Optional[float] = None
    required_below: Optional[float] = None
    val_set: Optional[ValSet] = None
    base_model: Optional[str] = None    # the base model every score was measured on

    @property
    def promote(self) -> bool:
        return self.outcome is GateOutcome.PROMOTE

    @property
    def promoted_checkpoint_id(self) -> Optional[str]:
        if not self.promote or self.best is None:
            return None
        return self.best.checkpoint_id

    @property
    def summary(self) -> str:
        return f"fidelity gate {self.outcome.value} ({self.reason.value}): {self.detail}"

    def to_record(self) -> Dict[str, Any]:
        """The durable decision record; every field present, null when missing."""
        best, final, baseline, val_set = self.best, self.final, self.baseline, self.val_set
        return {
            "schema": GATE_RECORD_SCHEMA,
            "decision": self.outcome.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "best_checkpoint_id": best.checkpoint_id if best else None,
            "best_iteration": best.iteration if best else None,
            "best_val_loss": _json_number(best.val_loss) if best else None,
            "final_checkpoint_id": final.checkpoint_id if final else None,
            "final_iteration": final.iteration if final else None,
            "final_val_loss": _json_number(final.val_loss) if final else None,
            # Every checkpoint the gate was given, with the score of its file.
            "checkpoints": [
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "iteration": checkpoint.iteration,
                    "val_loss": _json_number(checkpoint.val_loss),
                    "error": checkpoint.error,
                }
                for checkpoint in self.checkpoints
            ],
            "baseline_source": baseline.source if baseline else None,
            "baseline_val_loss": _json_number(baseline.val_loss) if baseline else None,
            "baseline_error": baseline.error if baseline else None,
            "baseline_changed": baseline.changed if baseline else None,
            "base_model": self.base_model,
            "margin": self.gate.margin,
            "required_below": _json_number(self.required_below),
            "overfit_check": self.gate.overfit_check,
            "overfit_ratio": self.gate.overfit_ratio,
            "ratio": _json_number(self.ratio),
            "val_set_digest": val_set.digest if val_set else None,
            "val_set_examples": val_set.examples if val_set else None,
            "val_set_batch_size": val_set.batch_size if val_set else None,
            "promoted_checkpoint_id": self.promoted_checkpoint_id,
        }


def _recorded_loss(value: Any, field: str) -> Optional[float]:
    """A loss as ``_json_number`` wrote it: a number, a non-finite string or null."""
    if value is None:
        return None
    if _is_real(value):
        return float(value)
    if isinstance(value, str):
        try:
            loss = float(value)
        except ValueError:
            pass
        else:
            if not math.isfinite(loss):
                return loss
    raise ValueError(f"{field}={value!r} is not a recorded loss")


def _recorded_text(value: Any, field: str) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{field}={value!r} is not text")


# The checkpoints, baseline, validation set and base model one gate run scored.
RecordedGateInputs = Tuple[Tuple[Checkpoint, ...], Optional[Baseline], Optional[ValSet], Optional[str]]


def recorded_inputs(record: Mapping[str, Any]) -> RecordedGateInputs:
    """The checkpoints, baseline, validation set and base model of one recorded gate run.

    The inverse of ``GateDecision.to_record`` for the gate's inputs, so a
    decision can be re-taken from the scores of one gate run. Raises ValueError
    when the record does not have that shape.
    """
    entries = record.get("checkpoints")
    if not isinstance(entries, list):
        raise ValueError("the record has no per-checkpoint scores")
    checkpoints = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"checkpoint entry {entry!r} is not a mapping")
        checkpoint_id, iteration = entry.get("checkpoint_id"), entry.get("iteration")
        if not isinstance(checkpoint_id, str) or not isinstance(iteration, int) or isinstance(iteration, bool):
            raise ValueError(f"checkpoint entry {dict(entry)!r} has no id and iteration")
        # An id names a weights file inside the candidate dir, nothing else.
        if PurePath(checkpoint_id).name != checkpoint_id or checkpoint_id in (".", ".."):
            raise ValueError(f"checkpoint id {checkpoint_id!r} is not a file name")
        checkpoints.append(Checkpoint(
            checkpoint_id, iteration,
            _recorded_loss(entry.get("val_loss"), f"{checkpoint_id} val_loss"),
            _recorded_text(entry.get("error"), f"{checkpoint_id} error"),
        ))
    source = _recorded_text(record.get("baseline_source"), "baseline_source")
    baseline = None if source is None else Baseline(
        source,
        _recorded_loss(record.get("baseline_val_loss"), "baseline_val_loss"),
        _recorded_text(record.get("baseline_error"), "baseline_error"),
        _recorded_text(record.get("baseline_changed"), "baseline_changed"),
    )
    digest, examples, batch_size = (
        record.get("val_set_digest"), record.get("val_set_examples"), record.get("val_set_batch_size"),
    )
    if digest is None and examples is None and batch_size is None:
        val_set = None
    elif isinstance(digest, str) and all(
        isinstance(count, int) and not isinstance(count, bool) for count in (examples, batch_size)
    ):
        val_set = ValSet(digest, examples, batch_size)
    else:
        raise ValueError(
            f"val_set_digest={digest!r} / val_set_examples={examples!r} / "
            f"val_set_batch_size={batch_size!r} is not a validation set"
        )
    base_model = _recorded_text(record.get("base_model"), "base_model")
    return tuple(checkpoints), baseline, val_set, base_model


@dataclass(frozen=True)
class FidelityGate:
    """Promote a run's best checkpoint only if it beats the baseline."""

    margin: float = 0.02            # relative: best < baseline * (1 - margin)
    overfit_ratio: float = 1.20     # reject when final / best exceeds this
    overfit_check: bool = True

    def __post_init__(self) -> None:
        if not _is_real(self.margin) or not math.isfinite(self.margin) or not 0.0 <= self.margin < 1.0:
            raise ValueError(f"gate.margin={self.margin!r} must be a number in [0, 1)")
        if (
            not _is_real(self.overfit_ratio)
            or not math.isfinite(self.overfit_ratio)
            or self.overfit_ratio < 1.0
        ):
            raise ValueError(
                f"gate.overfit_ratio={self.overfit_ratio!r} must be a finite number >= 1"
            )
        if not isinstance(self.overfit_check, bool):
            raise ValueError(f"gate.overfit_check={self.overfit_check!r} must be true or false")

    def to_config(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(
        cls, value: Any, *, base: Optional["FidelityGate"] = None,
    ) -> "FidelityGate":
        """Build a gate from a ``gate`` config mapping, overriding ``base``.

        Unknown keys and invalid values are refused, never ignored.
        """
        if not isinstance(value, Mapping):
            raise ValueError("gate config must be a mapping of margin/overfit_ratio/overfit_check")
        fields = (base or cls()).to_config()
        unknown = sorted(set(value) - set(fields))
        if unknown:
            raise ValueError(f"unknown gate config key(s): {', '.join(map(str, unknown))}")
        fields.update(value)
        return cls(**fields)

    def evaluate(
        self,
        *,
        checkpoints: Sequence[Checkpoint],
        baseline: Optional[Baseline],
        val_set: Optional[ValSet],
        base_model: Optional[str],
    ) -> GateDecision:
        """Decide whether this run's best checkpoint may be promoted.

        ``base_model`` is the base model every score was measured on (None
        when nothing was scored); the decision records it.
        """

        checkpoints = tuple(checkpoints)

        def decide(outcome: GateOutcome, reason: GateReason, detail: str, **found: Any) -> GateDecision:
            return GateDecision(
                outcome, reason, detail, self,
                checkpoints=checkpoints, baseline=baseline, val_set=val_set,
                base_model=base_model, **found,
            )

        def inconclusive(reason: GateReason, detail: str, **found: Any) -> GateDecision:
            return decide(GateOutcome.INCONCLUSIVE, reason, detail, **found)

        if val_set is None:
            return inconclusive(GateReason.VAL_SET_MISSING, "no validation set; nothing was scored")
        if val_set.examples <= 0:
            return inconclusive(GateReason.VAL_SET_EMPTY, "validation set is empty; nothing was scored")
        if val_set.examples < val_set.batch_size:
            return inconclusive(
                GateReason.VAL_SET_TOO_SMALL,
                f"validation set has {val_set.examples} example(s), fewer than one scoring batch "
                f"of {val_set.batch_size}; nothing was scored",
            )
        if not checkpoints:
            return inconclusive(GateReason.NO_CHECKPOINTS, "the run saved no checkpoints")

        final = max(checkpoints, key=lambda checkpoint: checkpoint.iteration)
        failed = [checkpoint for checkpoint in checkpoints if checkpoint.error is not None]
        if failed:
            return inconclusive(
                GateReason.CANDIDATE_EVAL_FAILED,
                f"checkpoint {failed[0].checkpoint_id} could not be scored: {failed[0].error}",
                final=final,
            )
        invalid = [
            checkpoint for checkpoint in checkpoints
            if checkpoint.val_loss is not None and not is_usable_loss(checkpoint.val_loss)
        ]
        if invalid:
            return inconclusive(
                GateReason.CANDIDATE_LOSS_INVALID,
                f"checkpoint {invalid[0].checkpoint_id} val loss is {_fmt(invalid[0].val_loss)}; "
                "a usable loss is finite and positive",
                final=final,
            )
        scored = [checkpoint for checkpoint in checkpoints if checkpoint.val_loss is not None]
        if not scored:
            return inconclusive(
                GateReason.NO_CHECKPOINT_VAL_LOSS,
                "no checkpoint has a validation loss, so there is no best checkpoint",
                final=final,
            )
        # Lowest loss wins; on a tie the earlier (less trained) checkpoint does.
        best = min(scored, key=lambda checkpoint: (checkpoint.val_loss, checkpoint.iteration))
        ratio = loss_ratio(final.val_loss, best.val_loss) if final.val_loss is not None else None
        found = {"best": best, "final": final, "ratio": ratio}
        if self.overfit_check and final.val_loss is None:
            return inconclusive(
                GateReason.FINAL_CHECKPOINT_NO_VAL_LOSS,
                f"final checkpoint {final.checkpoint_id} has no validation loss, so overfit "
                "cannot be checked",
                **found,
            )

        if baseline is None:
            return inconclusive(GateReason.BASELINE_UNAVAILABLE, "no baseline to compare against", **found)
        if baseline.changed is not None:
            return inconclusive(
                GateReason.BASELINE_CHANGED,
                f"baseline {baseline.source} is no longer the baseline: {baseline.changed}",
                **found,
            )
        if baseline.error is not None:
            return inconclusive(
                GateReason.BASELINE_EVAL_FAILED,
                f"baseline {baseline.source} could not be scored: {baseline.error}",
                **found,
            )
        if baseline.val_loss is None:
            return inconclusive(
                GateReason.BASELINE_UNAVAILABLE,
                f"baseline {baseline.source} has no validation loss on this validation set",
                **found,
            )
        if not is_usable_loss(baseline.val_loss):
            return inconclusive(
                GateReason.BASELINE_LOSS_INVALID,
                f"baseline {baseline.source} val loss is {_fmt(baseline.val_loss)}; "
                "a usable loss is finite and positive",
                **found,
            )
        required_below = margin_threshold(baseline.val_loss, self.margin)
        found["required_below"] = required_below

        if self.overfit_check and ratio > self.overfit_ratio:
            return decide(
                GateOutcome.REJECT, GateReason.OVERFIT,
                f"final checkpoint {final.checkpoint_id} val {_fmt(final.val_loss)} is "
                f"{ratio:.3f}x best {best.checkpoint_id} val {_fmt(best.val_loss)} "
                f"(limit {self.overfit_ratio:g}x): the run overfit",
                **found,
            )
        if not best.val_loss < required_below:
            return decide(
                GateOutcome.REJECT, GateReason.MARGIN,
                f"best checkpoint {best.checkpoint_id} val {_fmt(best.val_loss)} does not beat "
                f"{baseline.source} val {_fmt(baseline.val_loss)} by {self.margin * 100:g}% "
                f"(needs < {required_below:.3f})",
                **found,
            )
        return decide(
            GateOutcome.PROMOTE, GateReason.PASSED,
            f"best checkpoint {best.checkpoint_id} val {_fmt(best.val_loss)} beats "
            f"{baseline.source} val {_fmt(baseline.val_loss)} by more than {self.margin * 100:g}% "
            f"(needs < {required_below:.3f})"
            + (f"; final/best {ratio:.3f}x" if ratio is not None else ""),
            **found,
        )
