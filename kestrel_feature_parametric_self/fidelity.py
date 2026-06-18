"""Fidelity gate — decides whether a freshly trained adapter may be promoted.

This is the §5.2 promotion gate from docs/TWO_BRAIN_ARCHITECTURE.md: a nightly
adapter is only swapped in if it clears a held-out fidelity check. The concrete
metric here is validation loss (what ``mlx_lm.lora`` reports on the held-out
split), with two guards:

  - an absolute ceiling (``max_val_loss``) — reject an adapter that is simply bad;
  - a regression guard (``max_regression``) — reject an adapter meaningfully
    worse than the currently-served one, so the self can't drift downhill night
    over night.

Both the parser and the gate are pure functions so they are unit-testable
without MLX or a real training run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Matches mlx_lm.lora lines like: "Iter 400: Val loss 2.208, Val took 42.083s"
_VAL_LOSS_RE = re.compile(r"Val loss\s+([0-9]+(?:\.[0-9]+)?)")
# Matches the iteration counter in either Train- or Val-loss lines: "Iter 400:".
_ITER_RE = re.compile(r"Iter\s+([0-9]+)")


def parse_final_val_loss(log_text: str) -> Optional[float]:
    """Return the last validation loss reported in an mlx_lm.lora training log.

    Returns None if the log contains no validation loss (e.g. the run never
    reached an eval step or failed early) — which the gate treats as "cannot
    verify, do not promote".
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


@dataclass
class GateDecision:
    """Outcome of a fidelity evaluation."""

    promote: bool
    reason: str
    val_loss: Optional[float] = None


@dataclass
class FidelityGate:
    """Promote a new adapter only if its held-out loss clears the bar."""

    max_val_loss: float = 3.0       # absolute ceiling
    max_regression: float = 0.25    # allowed increase vs the served adapter

    def evaluate(
        self,
        new_val_loss: Optional[float],
        prior_val_loss: Optional[float] = None,
    ) -> GateDecision:
        """Decide whether ``new_val_loss`` may be promoted over ``prior_val_loss``."""
        if new_val_loss is None:
            return GateDecision(False, "no validation loss in training log — cannot verify", None)
        if new_val_loss > self.max_val_loss:
            return GateDecision(
                False,
                f"val loss {new_val_loss:.3f} exceeds ceiling {self.max_val_loss:.3f}",
                new_val_loss,
            )
        if prior_val_loss is not None and new_val_loss > prior_val_loss + self.max_regression:
            return GateDecision(
                False,
                (
                    f"val loss {new_val_loss:.3f} regressed vs served "
                    f"{prior_val_loss:.3f} (+{self.max_regression:.3f} allowed)"
                ),
                new_val_loss,
            )
        return GateDecision(True, f"val loss {new_val_loss:.3f} within bounds", new_val_loss)
