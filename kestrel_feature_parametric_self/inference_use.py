"""Whether a parametric-self adapter may influence live reasoning.

One switch covers every route by which an adapter could reach a turn:
in-loop serving, oracle consultation and conditioning
(``docs/TWO_BRAIN_ARCHITECTURE.md`` §4 and §10.2). None of those routes
exists in this release, so ``off`` is the only value. A value for a route
nobody has built would be a switch that gates nothing.

Adding a route means three things together: adding its value here, gating
the route on :meth:`ParametricSelfFeature.adapter_for_inference`, and, for
an agent whose consent to training was conditional on this switch (Meridian,
epic #1, 2026-10-20), that agent's recorded consent and the Sovereign's
approval before the value is set for it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class InferenceUse(str, Enum):
    """Routes by which an adapter may influence a live turn."""

    OFF = "off"


def parse_inference_use(value: Any) -> InferenceUse:
    """Return the switch value for ``value``, refusing anything not supported.

    Strict on purpose: only the exact string ``"off"`` (or the enum member) is
    accepted. ``False``, ``"OFF"`` or ``"no"`` are refused rather than read as
    off, so that a later value can never be reached by a spelling that this
    release happened to tolerate.
    """
    if isinstance(value, InferenceUse):
        return value
    if isinstance(value, str):
        try:
            return InferenceUse(value)
        except ValueError:
            pass
    raise ValueError(
        f"inference_use={value!r} is not supported: 'off' is the only value. "
        "No in-loop, oracle or conditioning path exists in this release."
    )
